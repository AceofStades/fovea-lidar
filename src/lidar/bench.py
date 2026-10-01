"""Benchmark: latency, memory and accuracy-by-distance on a SemanticKITTI sequence.

    python -m lidar.bench --checkpoint runs/spunet-5cm/best.pt --seq 08 --out docs/results

Writes results.json plus a markdown table. Without --checkpoint it benchmarks the mapping stages
on ground-truth labels only.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .grid import GridConfig, VariableResolutionGrid
from .labels import CATEGORY_NAMES, NUM_CATEGORIES, TRAIN_NAMES, UNKNOWN
from .metrics import IoU
from .pipeline import Pipeline


def timed(fn, sync=torch.cuda.synchronize):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/semantickitti")
    ap.add_argument("--seq", default="08")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-frames", type=int, default=0, help="limit to the first n frames (0 = all)")
    ap.add_argument("--compare-every", type=int, default=10, help="uniform-grid comparison every n-th frame")
    ap.add_argument("--accumulate", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="docs/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = Pipeline(args.root, args.seq, args.checkpoint, GridConfig(), args.accumulate)
    grid = pipe.grid
    uniform = VariableResolutionGrid(GridConfig(ratios=(1,), half_extents=(100.0,)), "cuda")
    frames = list(range(len(pipe) if not args.max_frames else min(args.max_frames, len(pipe))))
    stage_times = {}
    uniform_ms, ours_single_ms = [], []
    cell_agree = IoU(NUM_CATEGORIES)

    for n, i in enumerate(frames):
        out = pipe.step(i)  # consecutive frames, so temporal fusion behaves as in the simulator
        if n >= args.warmup:
            for k, v in out["times"].items():
                stage_times.setdefault(k, []).append(v)
            stage_times.setdefault("total", []).append(out["pipeline_ms"])

        if i % args.compare_every:
            continue
        # single-scan projection: ours vs a uniform 5 cm grid over the same 200 m x 200 m
        xyz = out["pts"][:, :3]
        cat = out["category"]
        _, t_ours = timed(lambda: grid.project(xyz, cat))
        _, t_uni = timed(lambda: uniform.project(xyz, cat))
        if n >= args.warmup:
            ours_single_ms.append(t_ours)
            uniform_ms.append(t_uni)

        # map fidelity: map from predictions vs map from ground-truth labels (same points)
        if out["gt"] is not None and pipe.source == "model":
            gt_cat = pipe._cat_lut[out["gt"].clamp_max(255)]
            f_pred = grid.project(xyz, cat)
            f_gt = grid.project(xyz, gt_cat)
            m = (f_gt.count > 0) & (f_gt.category != UNKNOWN)
            cell_agree.update(f_pred.category[m].long().cpu().numpy(), f_gt.category[m].long().cpu().numpy())

        if i % 500 == 0:
            print(f"frame {i}: {out['pipeline_ms']:.1f} ms", flush=True)

    mem = grid.memory_report()
    res = {
        "sequence": args.seq, "frames_evaluated": len(frames), "accumulate": args.accumulate,
        "source": pipe.source, "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(0),
        "latency_ms": {k: {"mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
                       for k, v in stage_times.items()},
        "single_scan_projection_ms": {"ours": float(np.mean(ours_single_ms)), "uniform_5cm": float(np.mean(uniform_ms))},
        "memory": mem,
    }
    res["fps_p50"] = 1000.0 / res["latency_ms"]["total"]["p50"]
    if pipe.source == "model":
        s = pipe.metrics_summary()
        res["segmentation"] = {
            "miou_19": s["miou_19"], "accuracy": s["acc"],
            "per_class_iou": dict(zip(TRAIN_NAMES, [None if np.isnan(v) else float(v) for v in pipe.iou.iou()])),
            "category_iou": s["category_iou"], "by_distance": s["by_distance"],
        }
        res["moving_objects"] = pipe.motion_summary()
        res["map_cells_vs_gt_map"] = {
            "cell_accuracy": cell_agree.accuracy(),
            "category_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(CATEGORY_NAMES, cell_agree.iou()) if n != "unknown"},
        }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
