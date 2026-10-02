"""Variable-resolution 2.5D grid engine.

Layout: a stack of concentric square rings ("clipmap"). Level k is a dense W_k x W_k grid with
cell size base_cell * ratio_k covering |x|, |y| < half_extent_k; a point is stored in the finest
level whose square contains it. Levels with the default config:

    L0   5 cm   |x|,|y| < 10 m
    L1  10 cm   10-20 m
    L2  20 cm   20-40 m
    L3  40 cm   40-100 m

No alignment error / no data loss, by construction:
  * every coordinate is first quantized to ONE integer lattice (base_cell); the ring a point
    belongs to and its cell index in that ring are both computed from the same integer, so
    there is no float disagreement at ring boundaries and each point lands in exactly one cell;
  * cell-size ratios are integers and every ring's inner edge is a multiple of the next
    coarser cell size, so a coarse cell is always an exact union of finer cells (the inner
    "hole" of level k is exactly level k-1, pooled), which lets statistics be pooled between
    levels without resampling.

Per cell we keep 2.5D layers: point count, ground height, obstacle top / clearance above ground,
semantic category (safety-priority rule), step height (curbs), moving-point count and a
traversability class.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .labels import (DRIVABLE, NUM_CATEGORIES, PEDESTRIAN, TERRAIN, UNKNOWN, VEGETATION)

# Traversability classes, in increasing severity
UNOBSERVED, FREE, ROUGH, OVERHANG, STEP, BLOCKED = range(6)
TRAVERSABILITY_NAMES = ["unobserved", "free", "rough", "overhang-passable", "step/curb", "blocked"]
TRAVERSABILITY_COLORS = np.array([
    [40, 44, 52], [46, 204, 113], [224, 184, 74], [124, 196, 255], [255, 140, 66], [255, 71, 87],
], dtype=np.uint8)

# What one cell costs when the map is stored / sent (also used for the uniform-grid baselines)
MAP_CELL_DTYPE = np.dtype([
    ("count", np.uint16), ("category", np.uint8), ("traversability", np.uint8),
    ("ground_z", np.float16), ("obstacle_top", np.float16), ("clearance", np.float16),
    ("moving", np.uint8),
])


@dataclass(frozen=True)
class GridConfig:
    base_cell: float = 0.05
    ratios: tuple = (1, 2, 4, 8)
    half_extents: tuple = (10.0, 20.0, 40.0, 100.0)
    z_range: tuple = (-5.0, 5.0)          # points outside are dropped (sensor frame)
    ground_prior_z: float = -1.73         # KITTI sensor height; used only if no ground is seen
    min_points_claim: int = 2             # obstacle category needs this many points to claim a cell
    min_obstacle_height: float = 0.25     # non-ground points lower than this are drive-over clutter
    geometric_obstacle_height: float = 0.5  # any point this high above ground is an obstacle
    vehicle_height: float = 2.0           # clearance needed to pass under an overhang
    step_threshold: float = 0.15          # height jump to a neighbour cell that counts as a curb


# Ring ladders. Any ladder works as long as cell-size ratios are integers and every inner edge is a
# multiple of the next coarser cell, so cells stay exactly nested.
PROFILES = {
    # 5 / 10 / 20 / 40 cm: resolution halves at each doubling of range
    "balanced": GridConfig(),
    # the problem statement's example: 5 cm within 10 m, coarsening to 50 cm out to 100 m
    "spec": GridConfig(ratios=(1, 2, 10), half_extents=(10.0, 20.0, 100.0)),
}


class VariableResolutionGrid:
    def __init__(self, cfg: GridConfig = GridConfig(), device="cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        n_half = [round(h / cfg.base_cell) for h in cfg.half_extents]
        for k, (h, n, r) in enumerate(zip(cfg.half_extents, n_half, cfg.ratios)):
            if abs(n * cfg.base_cell - h) > 1e-6 or n % r:
                raise ValueError(f"level {k}: half extent {h} m is not a multiple of its cell size")
            if k and (n_half[k - 1] % r or r % cfg.ratios[k - 1]):
                raise ValueError(f"level {k}: inner edge / ratio not aligned with level {k - 1}")
        self.n_half = n_half                                   # half extent in base cells
        self.widths = [2 * n // r for n, r in zip(n_half, cfg.ratios)]
        self.cell_sizes = [cfg.base_cell * r for r in cfg.ratios]
        sizes = [w * w for w in self.widths]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)]).tolist()
        self.num_cells = self.offsets[-1]
        self._n_half_t = torch.tensor(n_half, device=self.device)
        self._ratio_t = torch.tensor(cfg.ratios, device=self.device)
        self._offset_t = torch.tensor(self.offsets[:-1], device=self.device)
        self._width_t = torch.tensor(self.widths, device=self.device)

    # ------------------------------------------------------------------ geometry
    @property
    def num_levels(self):
        return len(self.widths)

    def hole(self, k):
        """Index range [a, b) of level k occupied by level k-1 (empty for k = 0)."""
        if k == 0:
            return 0, 0
        r = self.cfg.ratios[k]
        return (self.n_half[k] - self.n_half[k - 1]) // r, (self.n_half[k] + self.n_half[k - 1]) // r

    def used_cells(self):
        return sum(w * w - (lambda ab: (ab[1] - ab[0]) ** 2)(self.hole(k)) for k, w in enumerate(self.widths))

    def level_view(self, flat, k):
        w = self.widths[k]
        return flat[self.offsets[k]:self.offsets[k + 1]].view(w, w, *flat.shape[1:])

    def cell_index(self, xy):
        """(N, 2) metric xy -> (flat cell index, level); level == num_levels means out of range."""
        q = torch.floor(xy / self.cfg.base_cell).long()                  # one shared integer lattice
        cheb = torch.maximum(q, -q - 1).amax(1)                            # ring distance in base cells
        level = torch.searchsorted(self._n_half_t, cheb, right=True)
        lv = level.clamp_max(self.num_levels - 1)
        local = torch.div(q + self._n_half_t[lv, None], self._ratio_t[lv, None], rounding_mode="floor")
        flat = self._offset_t[lv] + local[:, 1] * self._width_t[lv] + local[:, 0]
        return flat, level

    def cell_centers(self, k):
        """(W, W, 2) metric xy of level k cell centers (row = y, col = x)."""
        w, s, h = self.widths[k], self.cell_sizes[k], self.cfg.half_extents[k]
        c = -h + (torch.arange(w, device=self.device) + 0.5) * s
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        return torch.stack([xx, yy], -1)

    # ------------------------------------------------------------------ projection
    @torch.no_grad()
    def project(self, xyz, category, moving=None):
        """Project classified points into the grid.

        xyz (N, 3) float, category (N,) map categories (labels.py), moving (N,) bool or None.
        Returns a GridFrame with flat per-cell layers.
        """
        cfg, dev, T = self.cfg, self.device, self.num_cells
        xyz = torch.as_tensor(xyz, device=dev, dtype=torch.float32)
        category = torch.as_tensor(category, device=dev).long()
        z = xyz[:, 2]
        flat, level = self.cell_index(xyz[:, :2])
        keep = (level < self.num_levels) & (z >= cfg.z_range[0]) & (z <= cfg.z_range[1])
        flat, z, category = flat[keep], z[keep], category[keep]
        moving = None if moving is None else torch.as_tensor(moving, device=dev)[keep]
        n_in = int(keep.sum())

        count = torch.zeros(T, dtype=torch.int32, device=dev).index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
        cat_counts = torch.zeros(T * NUM_CATEGORIES, dtype=torch.int32, device=dev)
        cat_counts.index_add_(0, flat * NUM_CATEGORIES + category, torch.ones_like(flat, dtype=torch.int32))
        cat_counts = cat_counts.view(T, NUM_CATEGORIES)

        # ground height: mean z of ground-class points, pooled into coarser holes, then hole-filled
        is_ground = (category == DRIVABLE) | (category == TERRAIN)
        g_sum = torch.zeros(T, device=dev).index_add_(0, flat[is_ground], z[is_ground])
        g_cnt = torch.zeros(T, device=dev).index_add_(0, flat[is_ground], torch.ones_like(z[is_ground]))
        ground_observed = g_cnt > 0
        ground_z = self._fill_ground(g_sum, g_cnt)

        # obstacle structure relative to the local ground
        h = z - ground_z[flat]
        is_obstacle = (~is_ground & (h > cfg.min_obstacle_height)) | (h > cfg.geometric_obstacle_height)
        clearance = torch.full((T,), float("inf"), device=dev).scatter_reduce_(
            0, flat[is_obstacle], h[is_obstacle], "amin")
        obstacle_top = torch.full((T,), float("-inf"), device=dev).scatter_reduce_(
            0, flat[is_obstacle], h[is_obstacle], "amax")
        moving_count = torch.zeros(T, dtype=torch.int32, device=dev)
        if moving is not None:
            moving_count.index_add_(0, flat[moving], torch.ones(int(moving.sum()), dtype=torch.int32, device=dev))

        category_map = self._decide_category(cat_counts, count)
        step = self._step_height(ground_z, ground_observed)
        trav = self._traversability(count, category_map, ground_observed, clearance, step)
        return GridFrame(self, count, cat_counts, category_map, ground_z, ground_observed,
                         clearance, obstacle_top, step, moving_count, trav, n_in)

    def _decide_category(self, cat_counts, count):
        """Safety-priority pooling: an obstacle-like category with >= min_points_claim points wins
        over any amount of ground (a pole is never averaged away inside a big road cell); among the
        rest, the majority wins."""
        prio = torch.arange(NUM_CATEGORIES, device=self.device)
        claimed = (cat_counts >= self.cfg.min_points_claim) & (prio >= VEGETATION)
        top_claim = (claimed * prio).amax(1)
        known = cat_counts[:, 1:]
        majority = torch.where(known.sum(1) > 0, known.argmax(1) + 1, torch.full_like(top_claim, UNKNOWN))
        out = torch.where(top_claim > 0, top_claim, majority)
        return torch.where(count > 0, out, torch.full_like(out, UNKNOWN)).to(torch.uint8)

    def _fill_ground(self, g_sum, g_cnt):
        """Ground height everywhere: each level's inner hole receives the (exactly nested) pooled
        statistics of the finer level, then gaps (under cars, behind walls) are filled with a
        weighted push-pull pyramid."""
        out = torch.empty_like(g_sum)
        prev_s = prev_c = None
        for k in range(self.num_levels):
            s, c = self.level_view(g_sum, k).clone(), self.level_view(g_cnt, k).clone()
            if k:
                a, b = self.hole(k)
                p = self.cfg.ratios[k] // self.cfg.ratios[k - 1]
                s[a:b, a:b] += _sum_pool(prev_s, p)
                c[a:b, a:b] += _sum_pool(prev_c, p)
            prev_s, prev_c = s, c
            filled = _push_pull(s, c)
            self.level_view(out, k).copy_(torch.nan_to_num(filled, nan=self.cfg.ground_prior_z))
        return out

    def _step_height(self, ground_z, observed):
        """Largest height jump to an observed 3x3 neighbour (curbs, pothole rims, walls of ditches)."""
        step = torch.zeros_like(ground_z)
        for k in range(self.num_levels):
            g = self.level_view(ground_z, k)[None, None]
            m = self.level_view(observed, k)[None, None]
            hi = F.max_pool2d(torch.where(m, g, torch.full_like(g, -1e4)), 3, 1, 1)
            lo = -F.max_pool2d(torch.where(m, -g, torch.full_like(g, -1e4)), 3, 1, 1)
            s = torch.maximum(hi - g, g - lo).clamp_min(0)
            self.level_view(step, k).copy_(torch.where(m, s, torch.zeros_like(s))[0, 0])
        return step

    def _traversability(self, count, category, ground_observed, clearance, step):
        cfg = self.cfg
        t = torch.full_like(category, UNOBSERVED)
        t = torch.where(count > 0, torch.full_like(t, FREE), t)
        t = torch.where((count > 0) & (category == TERRAIN), torch.full_like(t, ROUGH), t)
        has_obstacle = torch.isfinite(clearance)
        # an overhang only counts as passable if we actually saw the ground beneath it; a wall whose
        # lower part is occluded would otherwise look like something to drive under
        passable = has_obstacle & (clearance >= cfg.vehicle_height) & ground_observed
        t = torch.where(passable, torch.full_like(t, OVERHANG), t)
        t = torch.where(ground_observed & (step > cfg.step_threshold), torch.full_like(t, STEP), t)
        t = torch.where(has_obstacle & ~passable, torch.full_like(t, BLOCKED), t)
        return t

    # ------------------------------------------------------------------ memory accounting
    def memory_report(self, height_range=8.0):
        cfg = self.cfg
        side = 2 * cfg.half_extents[-1]
        uniform_cells = round(side / cfg.base_cell) ** 2
        voxels = uniform_cells * round(height_range / cfg.base_cell)
        b = MAP_CELL_DTYPE.itemsize
        return {
            "cell_bytes": b,
            "ours_cells_allocated": self.num_cells,
            "ours_cells_used": self.used_cells(),
            "ours_bytes": self.num_cells * b,
            "uniform_2p5d_cells": uniform_cells,
            "uniform_2p5d_bytes": uniform_cells * b,
            "dense_3d_voxels": voxels,
            "dense_3d_bytes_1B_per_voxel": voxels,
        }


@dataclass
class GridFrame:
    grid: VariableResolutionGrid
    count: torch.Tensor
    category_counts: torch.Tensor
    category: torch.Tensor
    ground_z: torch.Tensor
    ground_observed: torch.Tensor
    clearance: torch.Tensor
    obstacle_top: torch.Tensor
    step: torch.Tensor
    moving_count: torch.Tensor
    traversability: torch.Tensor
    num_points: int

    def level(self, name, k):
        return self.grid.level_view(getattr(self, name), k)

    def to_numpy(self):
        """Pack into the compact per-cell record that would be stored / transmitted."""
        out = np.zeros(self.grid.num_cells, dtype=MAP_CELL_DTYPE)
        out["count"] = self.count.clamp_max(65535).cpu().numpy()
        out["category"] = self.category.cpu().numpy()
        out["traversability"] = self.traversability.cpu().numpy()
        out["ground_z"] = self.ground_z.cpu().numpy()
        out["obstacle_top"] = torch.nan_to_num(self.obstacle_top, neginf=0).cpu().numpy()
        out["clearance"] = torch.nan_to_num(self.clearance, posinf=65504).cpu().numpy()
        out["moving"] = (self.moving_count > 0).cpu().numpy()
        return out


def _sum_pool(x, p):
    h, w = x.shape
    return x.view(h // p, p, w // p, p).sum((1, 3))


def _push_pull(s, c):
    """Weighted mean s / c with empty cells filled from progressively coarser averages."""
    h, w = s.shape
    if h <= 2 or w <= 2:
        total = c.sum()
        mean = torch.where(total > 0, s.sum() / total.clamp_min(1e-9), torch.full_like(total, float("nan")))
        return torch.where(c > 0, s / c.clamp_min(1e-9), mean)
    ph, pw = h % 2, w % 2
    s2 = _sum_pool(F.pad(s, (0, pw, 0, ph)), 2)
    c2 = _sum_pool(F.pad(c, (0, pw, 0, ph)), 2)
    up = _push_pull(s2, c2).repeat_interleave(2, 0).repeat_interleave(2, 1)[:h, :w]
    return torch.where(c > 0, s / c.clamp_min(1e-9), up)
