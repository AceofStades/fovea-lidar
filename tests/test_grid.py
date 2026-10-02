"""Grid engine invariants on a real scan: no point lost, exact nesting between levels."""
import numpy as np
import torch

from lidar.data import read_label, read_scan, scan_files
import pytest

from lidar.grid import PROFILES, GridConfig, VariableResolutionGrid, _sum_pool
from lidar.labels import train_to_category

dev = "cuda" if torch.cuda.is_available() else "cpu"


def _frame():
    scan, label = scan_files("data/semantickitti", ["08"])[0]
    pts = read_scan(scan)
    train, moving, _ = read_label(label)
    return pts, train_to_category(train), moving


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_in_range_point_lands_in_exactly_one_cell(profile):
    pts, cat, moving = _frame()
    g = VariableResolutionGrid(PROFILES[profile], dev)
    f = g.project(pts[:, :3], cat, moving)
    in_range = (np.abs(pts[:, :2]).max(1) < g.cfg.half_extents[-1]) & (pts[:, 2] >= -5) & (pts[:, 2] <= 5)
    assert f.num_points == in_range.sum()
    assert int(f.count.sum()) == in_range.sum()
    # nothing is ever written into a level's inner hole (that area belongs to the finer level)
    for k in range(1, g.num_levels):
        a, b = g.hole(k)
        assert int(f.level("count", k)[a:b, a:b].sum()) == 0


def test_boundary_points_are_assigned_consistently():
    g = VariableResolutionGrid(GridConfig(), dev)
    eps = 1e-4
    xy = torch.tensor([[10 - eps, 0], [10.0, 0], [-10.0, 0], [-10 - eps, 0], [99.99, 99.99], [100.0, 0]], device=dev)
    flat, level = g.cell_index(xy)
    assert level.tolist() == [0, 1, 0, 1, 3, 4]


def test_coarse_cells_are_exact_unions_of_fine_cells():
    """Projecting into a grid whose levels all use the coarse cell size gives the same counts as
    sum-pooling the fine level: no resampling / alignment error between levels."""
    pts, cat, _ = _frame()
    near = np.abs(pts[:, :2]).max(1) < 10
    fine = VariableResolutionGrid(GridConfig(ratios=(1,), half_extents=(10.0,)), dev)
    coarse = VariableResolutionGrid(GridConfig(base_cell=0.4, ratios=(1,), half_extents=(10.0,)), dev)
    cf = fine.project(pts[near, :3], cat[near]).level("count", 0).float()
    cc = coarse.project(pts[near, :3], cat[near]).level("count", 0).float()
    assert torch.equal(_sum_pool(cf, 8), cc)


def test_spec_profile_coarsens_to_50cm_at_100m():
    g = VariableResolutionGrid(PROFILES["spec"], dev)
    assert [round(c, 3) for c in g.cell_sizes] == [0.05, 0.1, 0.5]
    flat, level = g.cell_index(torch.tensor([[5.0, 0.0], [15.0, 0.0], [99.0, 0.0]], device=dev))
    assert level.tolist() == [0, 1, 2]


def test_misaligned_ladder_is_rejected():
    with pytest.raises(ValueError):
        VariableResolutionGrid(GridConfig(ratios=(1, 2, 5), half_extents=(10.0, 20.0, 100.0)), dev)  # 10 cm -> 25 cm
