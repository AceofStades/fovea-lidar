"""Strip a training checkpoint down to fp16 inference weights (optimizer state removed).

    python scripts/export_weights.py runs/kaggle/spunet-5cm-mix/run/best.pt models/spunet-5cm-mix.pt
"""
import sys
from pathlib import Path

import torch

src, dst = sys.argv[1], Path(sys.argv[2])
ck = torch.load(src, map_location="cpu", weights_only=False)
state = {k.removeprefix("module."): (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
keep = {k: ck["args"][k] for k in ("voxel", "width", "mix") if k in ck["args"]}
dst.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": state, "args": keep, "val_miou": ck.get("val_miou", ck.get("best")),
            "epoch": ck.get("epoch")}, dst)
print(f"{dst}: {dst.stat().st_size / 1e6:.1f} MB, val mIoU {ck.get('val_miou', ck.get('best')):.4f}, epoch {ck.get('epoch')}")
