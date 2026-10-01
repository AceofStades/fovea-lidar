"""End-to-end per-frame pipeline: scan -> segmentation -> temporal accumulation -> 2.5D grid ->
objects -> metrics. Used by the live simulator and by the benchmark script."""
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .data import read_label, read_lidar_poses, read_scan, scan_files
from .grid import GridConfig, VariableResolutionGrid
from .labels import (CATEGORY_NAMES, IGNORE, NUM_CATEGORIES, NUM_CLASSES, PEDESTRIAN, TRAIN_TO_CATEGORY,
                     UNKNOWN, VEHICLE)
from .metrics import IoU, IoUByDistance
from .objects import MotionCue, Tracker, detect


class Timer:
    def __init__(self, device):
        self.sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
        self.times = {}
        self.t = time.perf_counter()

    def lap(self, name):
        self.sync()
        now = time.perf_counter()
        self.times[name] = (now - self.t) * 1000
        self.t = now


class Pipeline:
    def __init__(self, root, seq, checkpoint=None, grid_cfg=GridConfig(), accumulate=10, device="cuda"):
        self.device = torch.device(device)
        self.root, self.seq = root, seq
        self.files = scan_files(root, [seq], require_labels=False)
        if not self.files:
            raise SystemExit(f"no scans found for sequence {seq} under {root}")
        self.poses = read_lidar_poses(root, seq)[:len(self.files)]
        self.grid = VariableResolutionGrid(grid_cfg, device)
        self.segmenter = None
        if checkpoint:
            from .infer import Segmenter
            self.segmenter = Segmenter(checkpoint, device)
        self.source = "model" if self.segmenter else "gt"
        self.accumulate = accumulate
        self.history = deque()
        self.tracker = Tracker()
        self.motion = MotionCue()
        self.prev_index = None
        self._cat_lut = torch.full((256,), UNKNOWN, dtype=torch.long, device=self.device)
        self._cat_lut[:NUM_CLASSES] = torch.as_tensor(TRAIN_TO_CATEGORY, device=self.device).long()
        self._cell_geometry()
        self.reset_metrics()

    # ------------------------------------------------------------------ helpers
    def _cell_geometry(self):
        """Per-cell center / size / level lookup tables (flat order), built once."""
        centers, sizes, levels = [], [], []
        for k in range(self.grid.num_levels):
            c = self.grid.cell_centers(k).reshape(-1, 2)
            centers.append(c)
            sizes.append(torch.full((len(c),), self.grid.cell_sizes[k], device=self.device))
            levels.append(torch.full((len(c),), k, dtype=torch.uint8, device=self.device))
        self.cell_xy, self.cell_size, self.cell_level = torch.cat(centers), torch.cat(sizes), torch.cat(levels)

    def reset_metrics(self):
        self.iou = IoU(NUM_CLASSES)
        self.cat_iou = IoU(NUM_CATEGORIES)
        self.iou_dist = IoUByDistance(NUM_CATEGORIES)
        self.latency_hist = deque(maxlen=300)
        self.motion_tp = self.motion_fp = self.motion_fn = 0

    def has_labels(self, i):
        return self.files[i][1] is not None and Path(self.files[i][1]).exists()

    def __len__(self):
        return len(self.files)

    # ------------------------------------------------------------------ main step
    def step(self, i):
        tm = Timer(self.device)
        scan_path, label_path = self.files[i]
        pts = torch.from_numpy(read_scan(scan_path)).to(self.device)
        gt = moving = None
        if self.has_labels(i):
            gt_np, moving_np, _ = read_label(label_path)
            gt, moving = torch.from_numpy(gt_np.astype(np.int64)).to(self.device), torch.from_numpy(moving_np).to(self.device)
        tm.lap("load")

        if self.source == "model" and self.segmenter is not None:
            pred, conf = self.segmenter(pts)
        else:
            pred = gt if gt is not None else torch.full((len(pts),), IGNORE, device=self.device)
            conf = torch.ones(len(pts), device=self.device)
        category = self._cat_lut[pred.clamp_max(255)]
        tm.lap("inference")

        # temporal accumulation: static structure from the last frames, re-expressed in the current
        # sensor frame through the ego poses; dynamic points only from the current scan
        if self.prev_index is None or i != self.prev_index + 1:
            self.history.clear()
            self.tracker.reset()
            self.motion.reset()
        self.prev_index = i
        pose = torch.as_tensor(self.poses[i], device=self.device, dtype=torch.float32)
        xyz = pts[:, :3]
        static = (category != VEHICLE) & (category != PEDESTRIAN) & (category != UNKNOWN)
        world = xyz[static] @ pose[:3, :3].T + pose[:3, 3]
        self.history.append((world, category[static]))
        while len(self.history) > max(self.accumulate, 1):
            self.history.popleft()
        inv = torch.linalg.inv(pose)
        parts_xyz, parts_cat = [xyz], [category]
        for w, c in list(self.history)[:-1]:
            parts_xyz.append(w @ inv[:3, :3].T + inv[:3, 3])
            parts_cat.append(c)
        all_xyz, all_cat = torch.cat(parts_xyz), torch.cat(parts_cat)
        tm.lap("accumulate")

        # objects + motion state from the tracker (works the same for network and ground-truth labels)
        xyz_np, cat_np = xyz.cpu().numpy(), category.cpu().numpy()
        boxes = detect(xyz_np, cat_np)
        world_all = xyz @ pose[:3, :3].T + pose[:3, 3]
        if boxes:
            idx = torch.from_numpy(np.concatenate([b["indices"] for b in boxes])).to(self.device)
            owner = torch.from_numpy(np.repeat(np.arange(len(boxes)), [len(b["indices"]) for b in boxes])).to(self.device)
            cue = self.motion.evaluate(world_all[idx], owner, len(boxes))
            if cue is not None:
                for b, ov, seen in zip(boxes, *cue):
                    b["overlap"], b["seen"] = float(ov), float(seen)
        dynamic = (category == VEHICLE) | (category == PEDESTRIAN)
        self.motion.push(world_all[dynamic], world_all[::4])
        boxes = self.tracker.update(boxes, self.poses[i])
        moving_np = np.zeros(len(xyz_np), dtype=bool)
        for b in boxes:
            if b["moving"]:
                moving_np[b.pop("indices")] = True
            else:
                b.pop("indices")
        mov = torch.zeros(len(all_xyz), dtype=torch.bool, device=self.device)
        mov[:len(moving_np)] = torch.from_numpy(moving_np).to(self.device)
        tm.lap("objects")

        frame = self.grid.project(all_xyz, all_cat, mov)
        tm.lap("grid")

        if gt is not None and self.source == "model":
            gt_np = gt.cpu().numpy()
            pred_np = pred.cpu().numpy()
            pred_np = np.where(pred_np == 255, 0, pred_np)
            self.iou.update(pred_np, gt_np)
            gt_cat = TRAIN_TO_CATEGORY[np.minimum(gt_np, NUM_CLASSES - 1)]
            gt_cat = np.where(gt_np == IGNORE, 255, gt_cat)
            self.cat_iou.update(cat_np.astype(np.int64), gt_cat)
            self.iou_dist.update(cat_np.astype(np.int64), gt_cat, xyz_np)
        if moving is not None:
            gm = moving.cpu().numpy()
            self.motion_tp += int((gm & moving_np).sum())
            self.motion_fp += int((~gm & moving_np).sum())
            self.motion_fn += int((gm & ~moving_np).sum())
        tm.lap("metrics")

        pipeline_ms = sum(v for k, v in tm.times.items() if k not in ("metrics",))
        self.latency_hist.append(pipeline_ms)
        return {
            "index": i, "pts": pts, "pred": pred, "gt": gt, "category": category, "conf": conf,
            "frame": frame, "boxes": boxes, "pose": self.poses[i], "times": tm.times,
            "pipeline_ms": pipeline_ms, "accumulated_points": int(len(all_xyz)),
        }

    def metrics_summary(self):
        if self.cat_iou.cm.sum() == 0:
            return {"motion_only": True, **self.motion_summary()} if self.motion_tp + self.motion_fn else None
        cat_iou = self.cat_iou.iou()
        return {
            "miou_19": self.iou.miou(),
            "acc": self.iou.accuracy(),
            "category_iou": {n: (None if np.isnan(v) else float(v)) for n, v in zip(CATEGORY_NAMES, cat_iou) if n != "unknown"},
            "by_distance": {k: {"miou": v[0], "acc": v[1], "points": v[2]} for k, v in self.iou_dist.summary().items()},
            "points_evaluated": int(self.iou.cm.sum()),
            **self.motion_summary(),
        }

    def motion_summary(self):
        tp, fp, fn = self.motion_tp, self.motion_fp, self.motion_fn
        return {"moving_iou": tp / max(tp + fp + fn, 1), "moving_precision": tp / max(tp + fp, 1),
                "moving_recall": tp / max(tp + fn, 1)}
