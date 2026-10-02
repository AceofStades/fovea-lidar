"""Train the sparse U-Net on SemanticKITTI.

Single GPU:   python -m lidar.train --root data/semantickitti --out runs/spunet
Multi GPU:    torchrun --nproc_per_node=2 -m lidar.train ...
Stops cleanly at --max-hours (Kaggle sessions end at 12 h) and resumes from <out>/last.pt.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .data import TRAIN_SEQS, VAL_SEQS, SemanticKittiVoxels, collate_voxels, read_label, scan_files
from .labels import IGNORE, NUM_CLASSES, TRAIN_NAMES
from .losses import seg_loss
from .metrics import IoU
from .models.spunet import build_model


def class_weights(root, seqs, samples=300):
    files = scan_files(root, seqs)
    idx = np.linspace(0, len(files) - 1, min(samples, len(files))).astype(int)
    counts = np.zeros(NUM_CLASSES)
    for i in idx:
        lab, _, _ = read_label(files[i][1])
        lab = lab[lab != IGNORE]
        counts += np.bincount(lab, minlength=NUM_CLASSES)[:NUM_CLASSES]
    freq = counts / counts.sum()
    w = 1.0 / np.sqrt(freq + 1e-4)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    meter = IoU(NUM_CLASSES)
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        logits = model(b["feats"].to(device), b["coords"].to(device), b["batch_size"])
        pred = logits.argmax(1)[b["inverse"].to(device)].cpu().numpy()
        meter.update(pred, b["point_labels"].numpy())
    model.train()
    return meter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/semantickitti")
    ap.add_argument("--out", default="runs/spunet")
    ap.add_argument("--train-seqs", default=",".join(TRAIN_SEQS))
    ap.add_argument("--val-seqs", default=",".join(VAL_SEQS))
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4, help="per GPU")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-step", type=int, default=1, help="use every n-th training frame")
    ap.add_argument("--val-step", type=int, default=20, help="use every n-th val frame during training")
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--max-iters", type=int, default=0, help="stop after n iterations (smoke tests)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--mix", type=float, default=0.0, help="probability of PolarMix augmentation")
    ap.add_argument("--width", type=float, default=1.0, help="channel width multiplier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh-files", action="store_true",
                    help="re-scan the dataset folder every epoch (train while a download is still running)")
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")
    main_proc = rank == 0
    # monotonic clock: does not advance while the machine is suspended, so a laptop going to sleep
    # cannot consume the time budget (and collapse the schedule) without training
    t_start = time.monotonic()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    train_set = SemanticKittiVoxels(args.root, args.train_seqs.split(","), args.voxel, train=True,
                                    step=args.train_step, mix=args.mix)
    val_set = SemanticKittiVoxels(args.root, args.val_seqs.split(","), args.voxel, train=False, step=args.val_step)
    sampler = DistributedSampler(train_set) if ddp else None
    train_loader = DataLoader(train_set, args.batch, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                              collate_fn=collate_voxels, drop_last=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, 1, num_workers=min(args.workers, 4), collate_fn=collate_voxels)
    if main_proc:
        print(f"train scans {len(train_set)}  val scans {len(val_set)}", flush=True)
    if not len(train_set) or not len(val_set):
        raise SystemExit(f"no labeled scans found under {args.root}")

    model = build_model(args.width, in_channels=5, num_classes=NUM_CLASSES).to(device)
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index or int(os.environ["LOCAL_RANK"])])
    core = model.module if ddp else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(train_loader)
    scaler = torch.amp.GradScaler()
    weights = class_weights(args.root, args.train_seqs.split(",")).to(device)

    def lr_at(progress, warmup=0.03):
        """Warmup then cosine; progress is whichever is further along: iterations or the time budget,
        so the schedule always anneals before a Kaggle session is killed."""
        if progress < warmup:
            return args.lr * max(progress / warmup, 0.01)
        return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / (1 - warmup)))))

    start_epoch, it, best, hours_before = 0, 0, 0.0, 0.0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, it, best, hours_before = ck["epoch"], ck["iter"], ck["best"], ck.get("hours", 0.0)
        if main_proc:
            print(f"resumed from epoch {start_epoch} iter {it} best {best:.4f} ({hours_before:.2f} h done)", flush=True)
    budget = args.max_hours + hours_before if args.max_hours < 1e8 else 1e9
    hours = torch.zeros(1, device=device)  # rank 0's clock, shared so all ranks stop on the same step

    def save(epoch, name="last.pt", **extra):
        if main_proc:
            torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "iter": it, "best": best, "hours": float(hours), "args": vars(args),
                        **extra}, os.path.join(args.out, name + ".tmp"))
            os.replace(os.path.join(args.out, name + ".tmp"), os.path.join(args.out, name))

    log = open(os.path.join(args.out, "log.jsonl"), "a") if main_proc else None
    stop = False
    for epoch in range(start_epoch, args.epochs):
        if args.refresh_files and not ddp:
            n_before = len(train_set)
            train_set.files = scan_files(args.root, args.train_seqs.split(","), require_labels=True, step=args.train_step)
            if len(train_set) != n_before:
                train_loader = DataLoader(train_set, args.batch, shuffle=True, num_workers=args.workers,
                                          collate_fn=collate_voxels, drop_last=True, pin_memory=True, persistent_workers=True)
                print(f"epoch {epoch}: training set now {len(train_set)} scans", flush=True)
        if sampler:
            sampler.set_epoch(epoch)
        t0 = time.monotonic()
        for b in train_loader:
            if it % 10 == 0:
                hours.fill_(hours_before + (time.monotonic() - t_start) / 3600)
                if ddp:
                    dist.broadcast(hours, 0)
            progress = max(it / total, float(hours) / budget)
            for g in opt.param_groups:
                g["lr"] = lr_at(progress)
            feats, coords, labels = b["feats"].to(device, non_blocking=True), b["coords"].to(device), b["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(feats, coords, b["batch_size"])
                loss, ce, lv = seg_loss(logits.float(), labels, weights, IGNORE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            it += 1
            if main_proc and it % args.log_every == 0:
                rate = args.log_every / (time.monotonic() - t0)
                t0 = time.monotonic()
                rec = {"iter": it, "epoch": epoch, "loss": loss.item(), "ce": ce.item(), "lovasz": lv.item(),
                       "lr": opt.param_groups[0]["lr"], "it_s": rate, "hours": float(hours), "progress": progress}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
            if progress >= 1.0 or (args.max_iters and it >= args.max_iters):
                stop = True
                break
        save(epoch + 1)
        if main_proc:
            meter = evaluate(core, val_loader, device)
            miou = meter.miou()
            rec = {"epoch": epoch + 1, "val_miou": miou, "val_acc": meter.accuracy(),
                   "per_class": dict(zip(TRAIN_NAMES, np.round(meter.iou(), 4).tolist()))}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if miou > best:
                best = miou
                save(epoch + 1, "best.pt", val_miou=miou)
                save(epoch + 1)  # keep 'best' in last.pt in sync
        if ddp:
            dist.barrier()
        if stop:
            break
    if main_proc:
        print(f"done: best val mIoU {best:.4f}, {(time.monotonic() - t_start) / 3600:.2f} h", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
