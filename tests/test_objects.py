"""Object clustering, tracking, motion cue and detection metrics on synthetic scenes."""
import numpy as np
import torch

from lidar.data import polarmix
from lidar.labels import NUM_CLASSES, PEDESTRIAN, TERRAIN, VEHICLE
from lidar.objects import MotionCue, Tracker, detect, plausible
from lidar.objeval import ObjectEval


def car(cx, cy, n=400, rng=np.random.default_rng(0)):
    """Points on the surface of a 4.2 x 1.8 x 1.5 m box resting on z = -1.73."""
    p = rng.uniform([-2.1, -0.9, 0], [2.1, 0.9, 1.5], size=(n, 3))
    p[:, 2] += -1.73
    p[:, :2] += [cx, cy]
    return p.astype(np.float32)


def test_two_separated_cars_give_two_plausible_boxes():
    xyz = np.concatenate([car(10, 0), car(10, 6)])
    cat = np.full(len(xyz), VEHICLE)
    boxes = detect(xyz, cat)
    assert len(boxes) == 2
    for b in boxes:
        assert plausible(b)
        assert 3.5 < max(b["size"][:2]) < 4.6
        assert len(b["indices"]) == 400


def test_flat_clutter_is_not_a_vehicle():
    rng = np.random.default_rng(1)
    xyz = rng.uniform([5, 5, -1.73], [6, 6, -1.6], size=(60, 3)).astype(np.float32)
    assert detect(xyz, np.full(len(xyz), VEHICLE)) == []


def test_tracker_measures_speed_in_world_frame():
    tr = Tracker()
    for k in range(10):
        pose = np.eye(4)
        pose[0, 3] = 1.0 * k                      # ego drives +1 m per frame (10 m/s)
        # object also drives at 10 m/s, so it stays at the same place in the sensor frame
        boxes = [{"center": [12.0, 0.0, 0.0], "category": VEHICLE}]
        out = tr.update(boxes, pose)
    assert abs(out[0]["speed"] - 10.0) < 0.5
    assert out[0]["moving"]


def test_parked_car_seen_from_a_moving_ego_is_static():
    tr = Tracker()
    for k in range(10):
        pose = np.eye(4)
        pose[0, 3] = 1.0 * k
        boxes = [{"center": [12.0 - k, 0.0, 0.0], "category": VEHICLE}]  # fixed in the world
        out = tr.update(boxes, pose)
    assert out[0]["speed"] < 0.5
    assert not out[0]["moving"]


def test_motion_cue_overlap():
    cue = MotionCue(lag=2)
    parked = torch.from_numpy(car(10, 0))
    for _ in range(3):
        cue.push(parked, parked)
    owner = torch.zeros(len(parked), dtype=torch.long)
    overlap, seen = cue.evaluate(parked, owner, 1)
    assert overlap[0] > 0.9 and seen[0] > 0.9
    moved = parked + torch.tensor([5.0, 0.0, 0.0])
    overlap, _ = cue.evaluate(moved, owner, 1)
    assert overlap[0] < 0.2


def test_object_eval_counts_matches_by_distance():
    xyz = np.concatenate([car(5, 0), car(25, 0)])
    train = np.zeros(len(xyz), np.uint8)                       # class 0 = car
    inst = np.repeat([1, 2], 400).astype(np.uint16)
    boxes = detect(xyz, np.full(len(xyz), VEHICLE))
    ev = ObjectEval()
    ev.update(xyz, train, inst, boxes, [b.pop("indices") for b in boxes])
    s = ev.summary()["vehicle"]
    assert s["objects"] == 2 and s["recall"] == 1.0 and s["precision"] == 1.0
    assert s["by_distance"]["0-10m"]["objects"] == 1 and s["by_distance"]["20-30m"]["objects"] == 1


def test_polarmix_keeps_labels_aligned():
    rng = np.random.default_rng(2)
    p1 = rng.normal(size=(1000, 4)).astype(np.float32)
    p2 = rng.normal(size=(800, 4)).astype(np.float32)
    l1 = np.full(1000, TERRAIN, np.uint8)
    l2 = np.where(rng.random(800) < 0.1, 5, 8).astype(np.uint8)  # some persons (rare class)
    p, l = polarmix(p1, l1, p2, l2)
    assert len(p) == len(l)
    assert (l == 5).sum() >= 3 * (l2 == 5).sum()  # rotated copies of rare points were pasted
    assert set(np.unique(l)) <= {TERRAIN, 5, 8}
    assert l.max() < NUM_CLASSES
