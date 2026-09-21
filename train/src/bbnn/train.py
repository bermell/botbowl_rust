"""Train the value/policy net on a prepared corpus: one board-dims directory,
or (plan 042) a directory of ``dims_*`` subdirs from a mixed-size corpus.

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

from .data import MultiDimsDataset, PreparedDataset, collate, make_loader, open_prepared
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


def compute_losses(model, batch, device, per_drive_value_weight=False):
    """Policy CE + value MSE.

    Plan 036 W4: with ``per_drive_value_weight`` the value MSE becomes
    ``(w * se).sum() / w.sum()`` with ``w = 1/len(drive)``, so a 60-position
    drive and a 10-position one contribute the same total value gradient — the
    label is one scalar per drive either way, and weighting by multiplicity is
    AlphaGo-2016's "one position per game" without discarding the other 29.
    Normalising by ``w.sum()`` rather than ``N`` keeps the loss on the same
    scale as the plain MSE, so ``--value-weight`` means the same thing with the
    flag on or off and the two arms' val curves stay directly comparable.
    """
    spatial = batch["spatial"].to(device)
    global_ = batch["global"].to(device)
    policy_out, value_out = model(spatial, global_)
    logits = masked_policy_logits(policy_out, batch["actions"].to(device), batch["pad_mask"].to(device))
    logsm = F.log_softmax(logits, dim=1)
    target = batch["policy"].to(device)                       # (N, K), sums to 1 per row
    policy_loss = -(target * logsm).sum(dim=1).mean()
    value_target = batch["value"].to(device)
    if per_drive_value_weight:
        w = batch["weight"].to(device)                        # (N, 1)
        value_loss = (w * (value_out - value_target) ** 2).sum() / w.sum()
    else:
        value_loss = F.mse_loss(value_out, value_target)
    pred = logits.argmax(dim=1)
    acc = (pred == batch["chosen"].to(device)).float().mean()
    return policy_loss, value_loss, acc


def evaluate(model, loader, device, per_drive_value_weight=False):
    """Mean policy loss / value MSE / top-1 over a held-out loader.

    Sample-weighted, not batch-weighted: a mixed corpus's per-group loader
    ends each group on a short batch, and weighting those equally with full
    ones would tilt the pooled number toward the smallest board.
    """
    model.eval()
    tot_p = tot_v = tot_a = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            pl, vl, acc = compute_losses(model, batch, device, per_drive_value_weight)
            b = batch["value"].shape[0]
            tot_p += pl.item() * b
            tot_v += vl.item() * b
            tot_a += acc.item() * b
            n += b
    return tot_p / n, tot_v / n, tot_a / n


def describe_data(tag, ds):
    """One line saying what a dataset holds — per board when mixed."""
    if isinstance(ds, MultiDimsDataset):
        parts = ", ".join(f"{k} {v}" for k, v in ds.group_sizes().items())
        return f"{tag}: {len(ds)} samples over {len(ds.groups)} board shapes ({parts})"
    return f"{tag}: {len(ds)} samples"


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
    max_steps=None,
    eval_every=None,
    width=64,
    blocks=6,
    value_weight=1.0,
    weight_decay=0.0,
    per_drive_value_weight=False,
):
    # Before anything that draws: the shuffle order, the augmentation flips,
    # and the weight init all come off global generators.
    if seed is not None:
        seed_everything(seed)
        print(f"seed: {seed}")

    # Plan 042: `dims_dir` may be one prepared dims dir or the parent of
    # several (a mixed-size corpus). Batches never mix board shapes.
    ds = open_prepared(dims_dir, augment=augment)
    print(describe_data("train", ds))
    if limit is not None:
        # Overfit smoke: restrict to the first `limit` samples.
        if isinstance(ds, MultiDimsDataset):
            raise ValueError("--limit is only supported on a single dims dir")
        ds.spatial = ds.spatial[:limit]
        ds.global_ = ds.global_[:limit]
        ds.value = ds.value[:limit]
        ds.chosen = ds.chosen[:limit]
        ds.offsets = ds.offsets[: limit + 1]
    loader = make_loader(ds, batch_size, shuffle=True)

    # Held-out set: must be prepared from *disjoint games* (hold out whole
    # generation shards) — samples within a game are consecutive states, so
    # a sample-level split leaks. No augmentation on the val pass.
    #
    # On a mixed corpus the pooled val numbers drive the restore, and each
    # board's own numbers are printed beside them: a size that lags is the
    # first thing plan 042's experiments need to see, and the pooled number
    # cannot show it.
    val_loader = None
    val_group_loaders = {}
    if val_dir is not None:
        val_ds = open_prepared(val_dir, augment=False)
        print(describe_data("val", val_ds))
        val_loader = make_loader(val_ds, batch_size, shuffle=False)
        if isinstance(val_ds, MultiDimsDataset):
            val_group_loaders = {
                name: DataLoader(g, batch_size=batch_size, shuffle=False, collate_fn=collate)
                for name, g in zip(val_ds.names, val_ds.groups)
            }

    device = resolve_device(device) if isinstance(device, str) else device
    # --init decides the architecture when given (a shape mismatch against
    # --width/--blocks would otherwise fail strict loading below); a fresh
    # net takes the requested size. 64x6 is the production default.
    if init is not None:
        state = torch.load(init, map_location=device)
        shape = BBNet.shape_of(state)
        if (shape["width"], shape["blocks"]) != (width, blocks):
            print(f"note: --init is {shape['width']}x{shape['blocks']}; building that, not {width}x{blocks}")
        model = BBNet(**shape).to(device)
    else:
        model = BBNet(width=width, blocks=blocks).to(device)
    print(f"model: width {model.stem.weight.shape[0]}, blocks {len(model.blocks)}, "
          f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

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
        model.load_state_dict(state)
        print(f"warm start: loaded weights ← {init}")

    # Plan 036 W2. AdamW's decay is decoupled, so at weight_decay=0 it is Adam
    # step for step — but stay on Adam anyway when the knob is off, so the
    # baseline arm is literally the code that ran before.
    #
    # BatchNorm scales/shifts and biases are excluded. Decaying a BN gamma
    # toward zero shrinks the activations it normalises and the tower just
    # relearns it in the next conv; decaying biases costs capacity for no
    # regularisation. This is the standard split and the model has BN
    # everywhere (`model.py`).
    if weight_decay:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
        )
        print(
            f"optimiser: AdamW lr {lr} weight_decay {weight_decay} "
            f"({len(decay)} decayed tensors, {len(no_decay)} exempt: BN + bias)"
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    if value_weight != 1.0:
        print(f"value loss weight: {value_weight} (restore criterion stays unweighted)")
    if per_drive_value_weight:
        print("value loss weighted per drive: 1/len(drive)")

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
        _, vv0, _ = evaluate(model, val_loader, device, per_drive_value_weight)
        print(f"epoch  -1  (warm-start baseline)  |  val_value {vv0:.4f}")

    # Step-driven when --max-steps is given, epoch-driven otherwise (production).
    #
    # Plan 029 needs this: comparing corpora of different sizes at a fixed epoch
    # count silently compares different amounts of gradient descent — a 7-
    # generation pool would take 7x the updates of a 1-generation pool, so "more
    # data wins" could just be "more updates wins". Fixing the step budget and
    # varying only how many unique samples those steps draw from is the whole
    # experiment. Re-iterating the DataLoader reshuffles, so an arm that makes 30
    # passes sees 30 different orders.
    step = 0
    epoch = 0
    stop = False
    # Validating once per epoch is fatal for the same comparison: at 110k steps a
    # 117k-sample pool yields 30 validation points and an 818k-sample pool yields
    # 4, so best-val restore would choose from candidate sets of wildly different
    # size — a confound inside the selection rule itself. A fixed step interval
    # gives every arm the same number of checkpoints.
    interval = eval_every if eval_every else None
    best_step = None
    # Plan 036's second, *reported* criterion: where val_policy alone bottoms
    # out. Nothing restores from it — it exists so the gap between the two
    # optima is a number on the training log rather than something to go
    # digging for. The plan's rule of thumb: if after W1-W4 the two still
    # differ by >5k steps, the heads want splitting at restore time.
    best_vp = None
    best_vp_step = None

    def validate(tag):
        """Val pass + best-checkpoint bookkeeping. Returns the printable suffix."""
        nonlocal best_val, best_epoch, best_step, best_state, best_vp, best_vp_step
        if val_loader is None:
            return ""
        vp, vv, va = evaluate(model, val_loader, device, per_drive_value_weight)
        # Per-board validation (plan 042). Printed on their own lines so the
        # pooled line — the one train_loop.sh greps — keeps its shape.
        for name, gl in val_group_loaders.items():
            gp, gv, ga = evaluate(model, gl, device, per_drive_value_weight)
            print(f"    val@{name}: val_policy {gp:.4f}  val_value {gv:.4f}  val_top1 {ga:.3f}", flush=True)
        model.train()
        if best_vp is None or vp < best_vp:
            best_vp, best_vp_step = vp, step
        suffix = f"  |  val_policy {vp:.4f}  val_value {vv:.4f}  val_top1 {va:.3f}"
        # Which head decides the restore. `value` is the historical rule and was
        # correct while the bot played `--evaluator nn-value`: priors came from
        # the scripted heuristic, so nothing consumed the policy head and
        # optimising it would have been optimising a dead output. Once the bot
        # plays `--evaluator nn` the policy head drives search, and restoring on
        # val_value alone actively discards it — plan 027 measured the two heads
        # saturating in completely different places (value by epoch 0-2, policy
        # still improving at epoch 9 in every generation).
        criterion = vv if select_on == "value" else vp + vv
        if best_val is None or criterion < best_val:
            best_val, best_epoch, best_step = criterion, epoch, step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out:  # persist immediately — a killed run keeps its best net
                torch.save(best_state, out)
            suffix += "  *"
        return suffix

    while not stop:
        if max_steps is None and epoch >= epochs:
            break
        model.train()
        tot_p = tot_v = tot_a = 0.0
        nb = 0
        for batch in loader:
            opt.zero_grad()
            pl, vl, acc = compute_losses(model, batch, device, per_drive_value_weight)
            (pl + value_weight * vl).backward()
            opt.step()
            step += 1
            tot_p += pl.item()
            tot_v += vl.item()
            tot_a += acc.item()
            nb += 1
            if interval and step % interval == 0:
                print(f"step {step:7d}  policy_loss {tot_p / nb:.4f}  "
                      f"value_loss {tot_v / nb:.4f}  top1_acc {tot_a / nb:.3f}" + validate("step"),
                      flush=True)
                tot_p = tot_v = tot_a = 0.0
                nb = 0
            if max_steps and step >= max_steps:
                stop = True
                break
        if nb and not interval:
            print(f"epoch {epoch:3d}  policy_loss {tot_p / nb:.4f}  "
                  f"value_loss {tot_v / nb:.4f}  top1_acc {tot_a / nb:.3f}" + validate("epoch"),
                  flush=True)
        epoch += 1
        if max_steps is None and epoch >= epochs:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        label = "val_value" if select_on == "value" else "val_policy+val_value"
        # Keep the literal prefix — train_loop.sh greps it. Report the step as
        # well as the epoch: with a fixed step budget the restored step is what
        # says whether a data-rich arm simply trained longer before overfitting,
        # which is a different claim from "more data taught it more" (plan 029).
        print(f"restored best-val weights: step {best_step} epoch {best_epoch} ({label} {best_val:.4f})")
        # Reported, never restored from — see `best_vp` above.
        if best_vp_step is not None:
            gap = best_vp_step - best_step
            print(
                f"policy-only optimum: step {best_vp_step} (val_policy {best_vp:.4f}), "
                f"{gap:+d} steps from the restore"
            )

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
    ap.add_argument(
        "--data",
        required=True,
        help="prepared board-dims dir (contains spatial.npy, ...), or a directory of dims_* subdirs "
             "from a mixed-board-size corpus (plan 042; batches never mix boards)",
    )
    ap.add_argument("--epochs", type=int, default=20, help="ignored when --max-steps is given")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="train for exactly N optimizer steps instead of N epochs. For comparing "
             "corpora of different sizes: at fixed epochs a bigger pool takes "
             "proportionally more gradient updates, so a fixed step budget is what "
             "isolates data volume from amount-of-descent (plan 029)",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="validate every N steps instead of once per epoch. Required with "
             "--max-steps: per-epoch validation gives a small pool many more "
             "checkpoints than a large one, putting a confound inside best-val restore",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="overfit only the first N samples")
    ap.add_argument("--no-augment", action="store_true", help="disable random y-flip augmentation")
    ap.add_argument(
        "--val-data",
        default=None,
        help="held-out prepared dims dir or directory of dims_* subdirs (disjoint games!); "
             "a mixed one also reports val loss per board",
    )
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
    ap.add_argument("--width", type=int, default=64, help="conv tower width (ignored with --init, which fixes the shape)")
    ap.add_argument("--blocks", type=int, default=6, help="residual blocks (ignored with --init)")
    ap.add_argument("--out", type=Path, default=None, help="save state_dict here")
    ap.add_argument("--onnx", type=Path, default=None, help="export ONNX here")
    ap.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="plan 036 W1: scale the value loss in the backward pass "
             "(policy_ce + W * value_mse). 1.0 is the historical unweighted sum; "
             "Leela Zero runs 0.25 after seeing exactly this value-head over-fit. "
             "The --select-on restore criterion stays *unweighted*: the weight is "
             "about gradient balance, the criterion measures the heads",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="plan 036 W2: AdamW decoupled weight decay (0.0 = plain Adam, as before). "
             "BatchNorm params and biases are exempt. AlphaZero's 1e-4 is an SGD L2 "
             "coefficient at lr 0.2, so it is a starting point here, not a translation",
    )
    ap.add_argument(
        "--per-drive-value-weight",
        action="store_true",
        help="plan 036 W4: weight each sample's value loss by 1/len(drive) from "
             "weight.npy, so every drive contributes one unit of value gradient "
             "per epoch instead of one per position. No effect on the policy loss",
    )
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
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        width=args.width,
        blocks=args.blocks,
        value_weight=args.value_weight,
        weight_decay=args.weight_decay,
        per_drive_value_weight=args.per_drive_value_weight,
    )


if __name__ == "__main__":
    main()
