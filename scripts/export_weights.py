"""Strip a training checkpoint down to fp16 inference weights (optimizer state removed).

    python scripts/export_weights.py runs/kaggle/spunet-5cm-mix/run/best.pt models/spunet-5cm-mix.pt \
        [docs/results/spunet-5cm-mix.json]

With a benchmark result file, the full sequence-08 mIoU is stored too (the training-time value is
measured on every 20th validation scan only).
"""
import json
import sys
from pathlib import Path

import torch

src, dst = sys.argv[1], Path(sys.argv[2])
ck = torch.load(src, map_location="cpu", weights_only=False)
state = {k.removeprefix("module."): (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
keep = {k: ck["args"][k] for k in ("voxel", "width", "mix") if k in ck["args"]}
dst.parent.mkdir(parents=True, exist_ok=True)
seq08 = json.load(open(sys.argv[3]))["segmentation"]["miou_19"] if len(sys.argv) > 3 else None
torch.save({"model": state, "args": keep, "val_miou": ck.get("val_miou", ck.get("best")),
            "seq08_miou": seq08, "epoch": ck.get("epoch")}, dst)
print(f"{dst}: {dst.stat().st_size / 1e6:.1f} MB, val mIoU {ck.get('val_miou', ck.get('best')):.4f}, epoch {ck.get('epoch')}")
