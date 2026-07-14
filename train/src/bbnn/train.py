"""Train the value/policy net on one prepared board-dims directory.

Loss = masked policy cross-entropy (per-sample log-softmax over the legal
action set) + value MSE. Logs total loss and chosen-action top-1 accuracy.
Overfitting a tiny subset should drive policy loss toward 0 — a plumbing
check, not a quality metric.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PreparedDataset, collate
from .export import export_onnx
from .model import BBNet, masked_policy_logits


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


def train(dims_dir, epochs=20, batch_size=32, lr=1e-3, limit=None, out=None, onnx=None, device="cpu", augment=True):
    ds = PreparedDataset(dims_dir, augment=augment)
    if limit is not None:
        # Overfit smoke: restrict to the first `limit` samples.
        ds.spatial = ds.spatial[:limit]
        ds.global_ = ds.global_[:limit]
        ds.value = ds.value[:limit]
        ds.chosen = ds.chosen[:limit]
        ds.offsets = ds.offsets[: limit + 1]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = BBNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

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
        print(
            f"epoch {epoch:3d}  policy_loss {tot_p / nb:.4f}  "
            f"value_loss {tot_v / nb:.4f}  top1_acc {tot_a / nb:.3f}"
        )

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
    ap.add_argument("--out", type=Path, default=None, help="save state_dict here")
    ap.add_argument("--onnx", type=Path, default=None, help="export ONNX here")
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
    )


if __name__ == "__main__":
    main()
