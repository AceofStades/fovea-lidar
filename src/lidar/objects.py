"""Object instances from segmented points, plus a light tracker for motion state.

Dynamic-category points (vehicles, pedestrians) are clustered by connected components on a
fine 2D occupancy grid; each cluster becomes an oriented box (PCA yaw). Boxes are associated
frame-to-frame in world coordinates (ego poses), which gives a velocity and lets us separate
moving traffic from parked cars without a dedicated motion network.

Motion evidence comes from two cues: (1) scan-to-scan overlap — a parked car's current points
land on voxels its own points occupied half a second ago, a moving car's do not; this is only
trusted where that area was actually observed back then; (2) a robust track velocity for objects
that are clearly fast. Cluster-centre velocity alone is unreliable because the visible side of a
car changes as we drive past it.
"""
from collections import deque

import numpy as np
import torch
from scipy import ndimage

from .labels import PEDESTRIAN, VEHICLE

MIN_POINTS = {VEHICLE: 12, PEDESTRIAN: 6}
CELL = {VEHICLE: 0.35, PEDESTRIAN: 0.25}
# plausible box shapes: (min height, max height, min footprint length, max footprint length) in metres
SHAPE = {VEHICLE: (0.5, 4.5, 0.8, 20.0), PEDESTRIAN: (0.5, 2.4, 0.0, 1.8)}


def plausible(box):
    lo_h, hi_h, lo_l, hi_l = SHAPE[box["category"]]
    length = max(box["size"][:2])
    return lo_h <= box["size"][2] <= hi_h and lo_l <= length <= hi_l


def detect(xyz, category):
    """xyz (N, 3) numpy, category (N,) numpy -> list of boxes (dicts) in the sensor frame, each with
    the indices of its points."""
    boxes = []
    for cat in (VEHICLE, PEDESTRIAN):
        sel = np.flatnonzero(category == cat)
        if len(sel) < MIN_POINTS[cat]:
            continue
        p = xyz[sel]
        ij = np.floor(p[:, :2] / CELL[cat]).astype(np.int64)
        ij -= ij.min(0)
        occ = np.zeros(ij.max(0) + 1, dtype=bool)
        occ[ij[:, 0], ij[:, 1]] = True
        lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
        point_lab = lab[ij[:, 0], ij[:, 1]]
        order = np.argsort(point_lab, kind="stable")
        bounds = np.searchsorted(point_lab[order], np.arange(1, n + 2))
        for k in range(n):
            idx = order[bounds[k]:bounds[k + 1]]
            if len(idx) < MIN_POINTS[cat]:
                continue
            box = _box(p[idx], cat)
            if not plausible(box):
                continue
            box["indices"] = sel[idx]  # into the input arrays; stripped before sending to clients
            boxes.append(box)
    return boxes


def _box(p, cat):
    xy = p[:, :2]
    c = xy.mean(0)
    if len(p) >= 5:
        evals, evecs = np.linalg.eigh(np.cov((xy - c).T))
        axis = evecs[:, np.argmax(evals)]
    else:
        axis = np.array([1.0, 0.0])
    yaw = float(np.arctan2(axis[1], axis[0]))
    rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    local = (xy - c) @ rot.T
    lo, hi = local.min(0), local.max(0)
    center = c + ((lo + hi) / 2) @ rot
    z0, z1 = float(p[:, 2].min()), float(p[:, 2].max())
    return {
        "category": int(cat), "center": [float(center[0]), float(center[1]), (z0 + z1) / 2],
        "size": [float(max(hi[0] - lo[0], 0.2)), float(max(hi[1] - lo[1], 0.2)), max(z1 - z0, 0.2)],
        "yaw": yaw, "points": int(len(p)), "distance": float(np.hypot(*center)),
    }


class MotionCue:
    """Voxel overlap between an object's current points and dynamic points from `lag` frames ago,
    plus how much of its footprint was observed at all back then (both in the world frame)."""

    def __init__(self, lag=5, voxel=0.5, seen_window=6):
        self.lag, self.voxel = lag, voxel
        self.dyn = deque(maxlen=lag)
        self.seen = deque(maxlen=lag + seen_window)

    def reset(self):
        self.dyn.clear()
        self.seen.clear()

    def _keys(self, w, dims):
        q = torch.floor(w[:, :dims] / self.voxel).long() + (1 << 20)
        k = q[:, 0]
        for d in range(1, dims):
            k = k * (1 << 21) + q[:, d]
        return k

    def evaluate(self, world_pts, box_index, num_boxes):
        """world_pts (M, 3) points that belong to boxes, box_index (M,) -> (overlap, seen) per box,
        or None while there is not enough history."""
        if len(self.dyn) < self.lag or num_boxes == 0:
            return None
        past = self.dyn[0]
        hit = torch.isin(self._keys(world_pts, 3), torch.unique(self._keys(past, 3))).float()
        old = list(self.seen)[:-self.lag + 1] if self.lag > 1 else list(self.seen)
        seen_keys = torch.unique(torch.cat([self._keys(w, 2) for w in old])) if old else torch.empty(0, dtype=torch.long, device=world_pts.device)
        seen = torch.isin(self._keys(world_pts, 2), seen_keys).float()
        n = torch.zeros(num_boxes, device=world_pts.device).index_add_(0, box_index, torch.ones_like(hit))
        overlap = torch.zeros(num_boxes, device=world_pts.device).index_add_(0, box_index, hit) / n.clamp_min(1)
        seen_frac = torch.zeros(num_boxes, device=world_pts.device).index_add_(0, box_index, seen) / n.clamp_min(1)
        return overlap.cpu().numpy(), seen_frac.cpu().numpy()

    def push(self, world_dynamic, world_all):
        self.dyn.append(world_dynamic)
        self.seen.append(world_all)


class Tracker:
    """Greedy nearest-neighbour association in the world frame. Velocity is the least-squares slope
    of the last few world positions, which is far less sensitive than frame-to-frame differences to
    the cluster centre shifting as the viewpoint changes."""

    def __init__(self, max_dist=2.5, fast_speed=4.0, dt=0.1, history=8, min_age=4,
                 overlap_moving=0.2, seen_min=0.5):
        self.tracks = {}
        self.next_id = 0
        self.max_dist, self.fast_speed, self.dt = max_dist, fast_speed, dt
        self.history, self.min_age = history, min_age
        self.overlap_moving, self.seen_min = overlap_moving, seen_min

    def reset(self):
        self.tracks, self.next_id = {}, 0

    def _velocity(self, hist):
        if len(hist) < 3:
            return np.zeros(2)
        t = np.arange(len(hist)) * self.dt
        p = np.array(hist)
        t = t - t.mean()
        return (t[:, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()

    def update(self, boxes, pose):
        """pose: 4x4 sensor-to-world of this frame. Adds 'id', 'speed', 'moving' to each box."""
        world = [pose[:2, :2] @ np.array(b["center"][:2]) + pose[:2, 3] for b in boxes]
        pairs = sorted(
            (np.linalg.norm(w - t["hist"][-1]), i, tid)
            for i, w in enumerate(world) for tid, t in self.tracks.items()
            if t["category"] == boxes[i]["category"])
        used, matched, new_tracks = set(), set(), {}
        for d, i, tid in pairs:
            if d > self.max_dist or i in used or tid in matched:
                continue
            used.add(i); matched.add(tid)
            t = self.tracks[tid]
            t["hist"] = (t["hist"] + [world[i]])[-self.history:]
            t["age"] += 1
            new_tracks[tid] = t
            boxes[i]["id"] = tid
        for i, b in enumerate(boxes):
            if i not in used:
                new_tracks[self.next_id] = {"hist": [world[i]], "age": 0, "category": b["category"], "evidence": 0.0}
                b["id"] = self.next_id
                self.next_id += 1
        self.tracks = new_tracks
        for b in boxes:
            t = self.tracks[b["id"]]
            speed = float(np.linalg.norm(self._velocity(t["hist"])))
            if b.get("seen") is not None and b["seen"] >= self.seen_min:
                moved = float(b["overlap"] < self.overlap_moving)
                t["evidence"] = 0.5 * t.get("evidence", 0.0) + 0.5 * moved if t["age"] else moved
            fast = t["age"] >= self.min_age and speed > self.fast_speed
            b["speed"] = speed
            b["moving"] = bool(fast or t.get("evidence", 0.0) > 0.5)
            b["age"] = t["age"]
        return boxes
