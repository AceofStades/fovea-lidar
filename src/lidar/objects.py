"""Object instances from segmented points, plus a light tracker for motion state.

Dynamic-category points (vehicles, pedestrians) are clustered by connected components on a
fine 2D occupancy grid; each cluster becomes an oriented box (PCA yaw). Boxes are associated
frame-to-frame in world coordinates (ego poses), which gives a velocity and lets us separate
moving traffic from parked cars without a dedicated motion network.
"""
import numpy as np
from scipy import ndimage

from .labels import PEDESTRIAN, VEHICLE

MIN_POINTS = {VEHICLE: 15, PEDESTRIAN: 6}
CELL = {VEHICLE: 0.35, PEDESTRIAN: 0.25}


def detect(xyz, category):
    """xyz (N, 3) numpy, category (N,) numpy -> list of boxes (dicts) in the sensor frame."""
    boxes = []
    for cat in (VEHICLE, PEDESTRIAN):
        sel = category == cat
        if sel.sum() < MIN_POINTS[cat]:
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
            boxes.append(_box(p[idx], cat))
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


class Tracker:
    """Greedy nearest-neighbour association in the world frame with an exponential velocity filter."""

    def __init__(self, max_dist=2.5, moving_speed=1.0, dt=0.1):
        self.tracks = {}
        self.next_id = 0
        self.max_dist, self.moving_speed, self.dt = max_dist, moving_speed, dt

    def reset(self):
        self.tracks, self.next_id = {}, 0

    def update(self, boxes, pose):
        """pose: 4x4 sensor-to-world of this frame. Adds 'id', 'speed', 'moving' to each box."""
        world = [pose[:2, :2] @ np.array(b["center"][:2]) + pose[:2, 3] for b in boxes]
        unmatched = set(self.tracks)
        pairs = sorted(
            (np.linalg.norm(w - t["pos"]), i, tid)
            for i, w in enumerate(world) for tid, t in self.tracks.items()
            if t["category"] == boxes[i]["category"])
        used = set()
        new_tracks = {}
        for d, i, tid in pairs:
            if d > self.max_dist or i in used or tid not in unmatched:
                continue
            used.add(i); unmatched.discard(tid)
            t = self.tracks[tid]
            vel = (world[i] - t["pos"]) / self.dt
            t["vel"] = 0.6 * t["vel"] + 0.4 * vel if t["age"] else vel
            t["pos"], t["age"] = world[i], t["age"] + 1
            new_tracks[tid] = t
            boxes[i]["id"] = tid
        for i, b in enumerate(boxes):
            if i not in used:
                new_tracks[self.next_id] = {"pos": world[i], "vel": np.zeros(2), "age": 0, "category": b["category"]}
                b["id"] = self.next_id
                self.next_id += 1
        self.tracks = new_tracks
        for b in boxes:
            t = self.tracks[b["id"]]
            speed = float(np.linalg.norm(t["vel"])) if t["age"] >= 2 else 0.0
            b["speed"] = speed
            b["moving"] = speed > self.moving_speed
            b["age"] = t["age"]
        return boxes
