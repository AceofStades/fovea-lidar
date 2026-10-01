"""Top-down raster of the variable-resolution grid (nearest-neighbour, so cell sizes stay visible)."""
import numpy as np
import torch

from .grid import TRAVERSABILITY_COLORS
from .labels import CATEGORY_COLORS


def composite(grid, frame_layer, half_extent, px, colors=None):
    """Sample a per-cell layer onto a px x px image covering |x|,|y| < half_extent (row 0 = +y)."""
    dev = frame_layer.device
    c = (torch.arange(px, device=dev) + 0.5) / px * 2 * half_extent - half_extent
    yy, xx = torch.meshgrid(c.flip(0), c, indexing="ij")
    flat, level = grid.cell_index(torch.stack([xx.reshape(-1), yy.reshape(-1)], 1))
    vals = frame_layer[flat.clamp_max(grid.num_cells - 1)]
    vals = torch.where(level < grid.num_levels, vals, torch.zeros_like(vals))
    img = vals.view(px, px).cpu().numpy()
    return colors[img] if colors is not None else img


def category_image(grid, frame, half_extent=100.0, px=1000):
    return composite(grid, frame.category.long(), half_extent, px, CATEGORY_COLORS)


def traversability_image(grid, frame, half_extent=100.0, px=1000):
    return composite(grid, frame.traversability.long(), half_extent, px, TRAVERSABILITY_COLORS)


def ring_outlines(grid, img, half_extent):
    px = img.shape[0]
    for h in grid.cfg.half_extents[:-1]:
        if h >= half_extent:
            continue
        a = int(round((half_extent - h) / (2 * half_extent) * px))
        b = px - a - 1
        img[a, a:b + 1] = img[b, a:b + 1] = img[a:b + 1, a] = img[a:b + 1, b] = 255
    return img
