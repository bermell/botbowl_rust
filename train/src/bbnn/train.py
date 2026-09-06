"""Train the value/policy net on one prepared board-dims directory.

Loss = masked policy cross-entropy (per-sample log-softmax over the legal
action set) + value MSE. Logs total loss and chosen-action top-1 accuracy.
Overfitting a tiny subset should drive policy loss toward 0 — a plumbing
check, not a quality metric.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PreparedDataset, collate
from .export import export_onnx
from .model import BBNet, masked_policy_logits


def seed_everything(seed):
    """Seed every RNG a training run draws from, so two runs on the same data
    take the same trajectory.

    Three consumers, all on global generators: `BBNet()`'s weight init,
    `DataLoader(shuffle=True)`'s permutation, and the per-access random y-flip
    augmentation in `data.py` (`torch.rand`). `random`/`numpy` are seeded too
    because they are one import away from being drawn on.

    This is deliberately *not* full determinism — no `use_deterministic_algorithms`,
    no cuDNN flags — because non-deterministic GPU kernel reductions cost nothing
    that matters here: what an A/B experiment needs is that its arms start from
    the same weights and see the same batch order, so a measured difference is
    attributable to the thing under test and not to the seed. Left unset,
    training is non-deterministic exactly as before.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec="auto"):
    """Map a --device spec to a torch device that can actually run kernels.

    `torch.cuda.is_available()` is not sufficient: a CUDA build whose kernels
    were compiled for newer architectures than the installed GPU still reports
    True, and only fails at the first kernel launch (the training host's
    GTX 1060 is sm_61; the PyPI cu130 wheel ships sm_75+). So we probe with a
    real launch and synchronize to surface the error here rather than mid-epoch.

    "auto" falls back to CPU with a printed reason; an explicit "cuda" raises,
    because silently training on CPU when the user asked for GPU is worse than
    stopping.
    """
    if spec == "cpu":
        return torch.device("cpu")

    want = "cuda" if spec == "auto" else spec
    if not want.startswith("cuda"):
        return torch.device(want)

    def _fail(reason):
        if spec == "auto":
            print(f"device: falling back to cpu ({reason})")
            return torch.device("cpu")
        raise RuntimeError(f"--device {spec} requested but unusable: {reason}")

    if not torch.cuda.is_available():
        return _fail("torch.cuda.is_available() is False")
    try:
        dev = torch.device(want)
        probe = torch.zeros(8, 8, device=dev)
        (probe @ probe).sum().item()
        torch.cuda.synchronize(dev)
    except Exception as e:  # noqa: BLE001 - any launch failure means unusable
        first = (str(e).splitlines() or [""])[0]
        return _fail(f"{type(e).__name__}: {first}")
    name = torch.cuda.get_device_name(dev)
    cc = ".".join(str(x) for x in torch.cuda.get_device_capability(dev))
    print(f"device: {dev} ({name}, sm_{cc.replace('.', '')})")
    return dev


def compute_losses(model, batch, device):
    spatial = batch["spatial"].to(device)
    global_ = batch["global"].to(device)
    policy_out, value_out = model(spatial, global_)
    logits = masked_policy_logits(policy_out, batch["actions"].to(device), batch["pad_mask"].to(device))
    logsm = F.log_softmax(logits, dim=1)
    target = batch["policy"].to(device)                       # (N, K), sums to 1 per row
    policy_loss = -(target * logsm).sum(dim=1).mean()
    value_loss = F.mse_loss(value_out, batch["value"].to(device))
    pred = logits.argmax(dim=1)
    acc = (pred == batch["chosen"].to(device)).float().mean()
    return policy_loss, value_loss, acc


def evaluate(model, loader, device):
    """Mean policy loss / value MSE / top-1 over a held-out loader."""
    model.eval()
    tot_p = tot_v = tot_a = 0.0
    nb = 0
    with torch.no_grad():
        for batch in loader:
            pl, vl, acc = compute_losses(model, batch, device)
            tot_p += pl.item()
            tot_v += vl.item()
            tot_a += acc.item()
            nb += 1
    return tot_p / nb, tot_v / nb, tot_a / nb


def train(
    dims_dir,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    limit=None,
    out=None,
    onnx=None,
    device="auto",
    augment=True,
    val_dir=None,
    init=None,
    select_on="value",
    seed=None,
):
    # Before anything that draws: the shuffle order, the augmentation flips,
    # and the weight init all come off global generators.
    if seed is not None:
        seed_everything(seed)
        print(f"seed: {seed}")

    ds = PreparedDataset(dims_dir, augment=augment)
    if limit is not None:
        # Overfit smoke: restrict to the first `limit` samples.
        ds.spatial = ds.spatial[:limit]
        ds.global_ = ds.global_[:limit]
        ds.value = ds.value[:limit]
        ds.chosen = ds.chosen[:limit]
        ds.offsets = ds.offsets[: limit + 1]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Held-out set: must be prepared from *disjoint games* (hold out whole
    # generation shards) — samples within a game are consecutive states, so
    # a sample-level split leaks. No augmentation on the val pass.
    val_loader = None
    if val_dir is not None:
        val_loader = DataLoader(
            PreparedDataset(val_dir, augment=False),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )

    device = resolve_device(device) if isinstance(device, str) else device
    model = BBNet().to(device)

    # Warm start (AlphaGo Zero keeps one continuously-trained net; generations
    # are checkpoints of a single SGD run, not independent retrainings). We
    # approximate that by seeding each generation from the champion's weights.
    # strict=True on purpose: a shape mismatch means the architecture moved
    # under us, and silently training a half-initialised net is worse than
    # stopping. Note this restores weights only, *not* Adam's moment
    # estimates — `out` must stay a bare state_dict because nn_server.py
    # loads it directly. Fresh moments at the usual 1e-3 would take large
    # first steps and undo the warm start, so callers should pass a lower
    # --lr when using --init (train_loop.sh does).
    if init is not None:
        model.load_state_dict(torch.load(init, map_location=device))
        print(f"warm start: loaded weights ← {init}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping via best-checkpoint restore: the value head starts
    # memorizing trajectories within a handful of epochs (plan 020 probe:
    # val optimum at epoch 3–6 while train loss keeps falling), so we keep
    # the weights from the best val value-MSE epoch and restore them at
    # the end rather than trusting the final epoch.
    best_val = None
    best_epoch = None
    best_state = None

    # Score the starting point before any training. On a warm start this is
    # the champion's own val_value on this corpus — the number every epoch
    # below has to beat for the generation to have been worth running. It is
    # deliberately *not* eligible for best-val restore: "restore the champion
    # unchanged" is a no-op candidate that would burn a full eval phase to
    # score 0.5 against itself.
    if val_loader is not None and init is not None:
        _, vv0, _ = evaluate(model, val_loader, device)
        print(f"epoch  -1  (warm-start baseline)  |  val_value {vv0:.4f}")

    for epoch in range(epochs):
        model.train()
        tot_p = tot_v = tot_a = 0.0
        nb = 0
        for batch in loader:
            opt.zero_grad()
            pl, vl, acc = compute_losses(model, batch, device)
            (pl + vl).backward()
            opt.step()
            tot_p += pl.item()
            tot_v += vl.item()
            tot_a += acc.item()
            nb += 1
        line = (
            f"epoch {epoch:3d}  policy_loss {tot_p / nb:.4f}  "
            f"value_loss {tot_v / nb:.4f}  top1_acc {tot_a / nb:.3f}"
        )
        if val_loader is not None:
            vp, vv, va = evaluate(model, val_loader, device)
            line += f"  |  val_policy {vp:.4f}  val_value {vv:.4f}  val_top1 {va:.3f}"
            # Which head decides the restore. `value` is the historical rule and
            # was correct while the bot played `--evaluator nn-value`: priors came
            # from the scripted heuristic, so nothing consumed the policy head and
            # optimising it would have been optimising a dead output. Once the bot
            # plays `--evaluator nn` the policy head drives search, and restoring
            # on val_value alone actively discards it — plan 027 measured the two
            # heads saturating in completely different places (value by epoch 0-2,
            # policy still improving at epoch 9 in every generation).
            #
            # `combined` uses the same sum the training loss minimises. Checked
            # against gen01-03: it picks epoch 2 for gen01 and gen02 (identical to
            # the value-only rule) and epoch 1 rather than 0 for gen03 — a small
            # change today that stops the rule ignoring a head that now matters.
            criterion = vv if select_on == "value" else vp + vv
            if best_val is None or criterion < best_val:
                best_val, best_epoch = criterion, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                # Persist immediately — a killed run keeps its best net.
                if out:
                    torch.save(best_state, out)
                line += "  *"
        print(line)

    if best_state is not None:
        model.load_state_dict(best_state)
        label = "val_value" if select_on == "value" else "val_policy+val_value"
        print(f"restored best-val weights: epoch {best_epoch} ({label} {best_val:.4f})")

    # Export/serialize from CPU: `export_onnx` traces with CPU dummy inputs,
    # and a .pt of CUDA tensors would pin the checkpoint to a GPU host.
    model = model.to("cpu")

    if out:
        torch.save(model.state_dict(), out)
        print(f"saved weights → {out}")
    if onnx:
        export_onnx(model, onnx)
        print(f"exported ONNX → {onnx}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Train the Blood Bowl value/policy net.")
    ap.add_argument("--data", required=True, help="prepared board-dims dir (contains spatial.npy, ...)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="overfit only the first N samples")
    ap.add_argument("--no-augment", action="store_true", help="disable random y-flip augmentation")
    ap.add_argument("--val-data", default=None, help="held-out prepared dims dir (disjoint games!)")
    ap.add_argument(
        "--select-on",
        choices=["value", "combined"],
        default="value",
        help="best-val restore criterion: `value` (val_value alone, the historical rule, "
             "correct only while the bot plays --evaluator nn-value and nothing consumes "
             "the policy head) or `combined` (val_policy + val_value, the training objective)",
    )
    ap.add_argument(
        "--init",
        type=Path,
        default=None,
        help="warm start: load this .pt state_dict before training (pass a lower --lr with it)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed python/numpy/torch (default: unseeded, as before). Exists for A/B experiments: "
             "weight init, DataLoader shuffle and the random y-flip augmentation are otherwise "
             "different every run, so two arms that should differ only in their data also differ "
             "by the seed, and a small measured gap cannot be attributed to either",
    )
    ap.add_argument("--out", type=Path, default=None, help="save state_dict here")
    ap.add_argument("--onnx", type=Path, default=None, help="export ONNX here")
    ap.add_argument(
        "--device",
        default="auto",
        help="auto (default; cuda if it can actually run kernels, else cpu), cpu, cuda, cuda:N",
    )
    args = ap.parse_args()
    train(
        args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
        out=args.out,
        onnx=args.onnx,
        augment=not args.no_augment,
        val_dir=args.val_data,
        device=args.device,
        init=args.init,
        select_on=args.select_on,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
