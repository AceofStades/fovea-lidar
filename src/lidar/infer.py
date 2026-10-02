"""Run a trained sparse U-Net on raw scans, entirely on the GPU."""
import torch

from .labels import NUM_CLASSES
from .models.spunet import build_model


class Segmenter:
    def __init__(self, checkpoint, device="cuda"):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.voxel = ck["args"]["voxel"]
        self.max_range = 100.0
        self.device = torch.device(device)
        model = build_model(ck["args"].get("width", 1.0), in_channels=5, num_classes=NUM_CLASSES)
        state = {k.removeprefix("module."): v for k, v in ck["model"].items()}
        model.load_state_dict(state)
        # spconv on Blackwell: fp16 inference works with a half() model, not with autocast
        self.model = model.to(self.device).eval().half()
        # full sequence-08 mIoU when the weights were exported with benchmark results
        self.val_miou = ck.get("seq08_miou") or ck.get("val_miou", ck.get("best"))

    @torch.no_grad()
    def __call__(self, pts):
        """pts (N, 4) tensor/array [x, y, z, remission] -> (train ids (N,), confidence (N,)).
        Points outside max_range get id 255."""
        pts = torch.as_tensor(pts, device=self.device, dtype=torch.float32)
        keep = pts[:, :2].abs().amax(1) < self.max_range
        p = pts[keep]
        q = torch.floor(p[:, :3] / self.voxel).long()
        q -= q.amin(0)
        dims = q.amax(0) + 1
        key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        uniq, inverse = torch.unique(key, return_inverse=True)
        rep = torch.zeros(len(uniq), dtype=torch.long, device=self.device).scatter_reduce_(
            0, inverse, torch.arange(len(p), device=self.device), "amax", include_self=False)
        coords = torch.cat([torch.zeros(len(uniq), 1, dtype=torch.int32, device=self.device), q[rep].int()], 1)
        xyz = p[rep, :3]
        feats = torch.cat([xyz, p[rep, 3:4], xyz[:, :2].norm(dim=1, keepdim=True) / 50.0], 1).half()
        logits = self.model(feats, coords, 1).float()
        prob = logits.softmax(1)
        conf, cls = prob.max(1)
        labels = torch.full((len(pts),), 255, dtype=torch.long, device=self.device)
        confidence = torch.zeros(len(pts), device=self.device)
        labels[keep] = cls[inverse]
        confidence[keep] = conf[inverse]
        return labels, confidence
