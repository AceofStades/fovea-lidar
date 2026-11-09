"""Record a stretch of a drive through the full pipeline, for the static web replay.

Runs exactly what the live simulator runs (segmentation -> fusion -> objects -> 2.5D grid) and saves,
per frame, what the simulator would send: the cell instances, the header (boxes, timings, metrics)
and the recent trajectory. `scripts/pack_replay.py` then compresses them for the browser.

    uv run python scripts/record_replay.py --checkpoint models/spunet-10cm-mix.pt --start 800 --frames 150
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lidar.grid import PROFILES
from lidar.pipeline import Pipeline
from lidar.grid import TRAVERSABILITY_COLORS, TRAVERSABILITY_NAMES
from lidar.labels import CATEGORY_COLORS, CATEGORY_NAMES
from lidar.sim.server import cell_instances  # the live simulator's own encoding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/semantickitti")
    ap.add_argument("--seq", default="08")
    ap.add_argument("--checkpoint", default="models/spunet-10cm-mix.pt")
    ap.add_argument("--accumulate", type=int, default=10)
    ap.add_argument("--start", type=int, default=800)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--out", default="runs/replay")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(args.root, args.seq, args.checkpoint, PROFILES["balanced"], args.accumulate, anchor="world")
    # warm up (kernel tuning), then fill the fusion window so the first recorded frame is complete
    for i in range(max(0, args.start - args.accumulate - 3), args.start):
        pipe.step(i)
    pipe.reset_metrics()

    for k in range(args.frames):
        i = args.start + k
        o = pipe.step(i)
        geo, attrs = cell_instances(pipe, o["frame"])
        inv = np.linalg.inv(o["pose"])
        lo = max(0, i - 80)
        traj = (inv @ pipe.poses[lo:i + 1, :, 3].T).T[:, :3].astype(np.float32)
        header = {
            "index": i, "frames": len(pipe), "source": pipe.source,
            "times": {kk: round(v, 3) for kk, v in o["times"].items()},
            "pipeline_ms": round(o["pipeline_ms"], 3),
            "boxes": o["boxes"],
            "num_points": int(len(o["pts"])),
            "accumulated_points": o["accumulated_points"],
            "observed_cells": int(len(geo)),
            "cells_per_level": np.bincount(attrs[:, 2], minlength=pipe.grid.num_levels).tolist(),
            "metrics": pipe.metrics_summary(),
            "sensor_from_map": o["sensor_from_map"].T.reshape(-1).tolist(),
        }
        np.savez(out / f"{k:04d}.npz", geo=geo, attrs=attrs, traj=traj)
        (out / f"{k:04d}.json").write_text(json.dumps(header, default=float))
        print(f"frame {i}: {len(geo)} cells, {o['pipeline_ms']:.1f} ms", flush=True)

    # the simulator's static info message, as the viewer expects it
    info = {
        "type": "info", "sequence": pipe.seq, "frames": len(pipe), "source": pipe.source,
        "has_model": pipe.segmenter is not None,
        "model_val_miou": getattr(pipe.segmenter, "val_miou", None),
        "model_voxel": getattr(pipe.segmenter, "voxel", None),
        "levels": [{"cell": s, "half_extent": h} for s, h in zip(pipe.grid.cell_sizes, pipe.grid.cfg.half_extents)],
        "memory": pipe.grid.memory_report(),
        "categories": CATEGORY_NAMES, "category_colors": CATEGORY_COLORS.tolist(),
        "traversability": TRAVERSABILITY_NAMES, "traversability_colors": TRAVERSABILITY_COLORS.tolist(),
        "vehicle_height": pipe.grid.cfg.vehicle_height,
        "accumulate": pipe.accumulate, "anchor": pipe.anchor,
        "replay": {"start": args.start, "frames": args.frames},
    }
    (out / "info.json").write_text(json.dumps(info, default=float))


if __name__ == "__main__":
    main()
