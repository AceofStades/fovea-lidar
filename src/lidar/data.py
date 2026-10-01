"""SemanticKITTI file access and the training dataset."""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .labels import IGNORE, raw_is_moving, raw_to_train

TRAIN_SEQS = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQS = ["08"]

DEFAULT_ROOT = os.environ.get("SEMKITTI_ROOT", "data/semantickitti")


def read_scan(path) -> np.ndarray:
    """(N, 4) float32: x, y, z, remission in the sensor frame."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_label(path):
    """Returns (train ids uint8, moving mask bool, instance ids uint16)."""
    raw = np.fromfile(path, dtype=np.uint32)
    return raw_to_train(raw), raw_is_moving(raw), (raw >> 16).astype(np.uint16)


@lru_cache(maxsize=None)
def _sequence_dirs(root):
    """All '.../sequences' folders under root. The walk never descends into a sequences folder,
    so it stays fast on the 60k-file Kaggle mount."""
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == "sequences":
            found.append(Path(dirpath))
            dirnames[:] = []
        elif dirpath.count(os.sep) - str(root).count(os.sep) >= 8:
            dirnames[:] = []
    return sorted(found)


def _find_seq_dir(root, seq, leaf):
    """Locate .../sequences/<seq>/<leaf> anywhere under root (local copy and the Kaggle copy
    keep velodyne and labels under different top-level folders)."""
    for s in _sequence_dirs(str(root)):
        if (s / seq / leaf).is_dir():
            return s / seq / leaf
    return None


def scan_files(root=DEFAULT_ROOT, seqs=VAL_SEQS, require_labels=True, step=1):
    """List (scan_path, label_path) pairs that exist on disk, in sequence/frame order."""
    pairs = []
    for seq in seqs:
        velo, labels = _find_seq_dir(root, seq, "velodyne"), _find_seq_dir(root, seq, "labels")
        if velo is None:
            continue
        for scan in sorted(velo.glob("*.bin"))[::step]:
            label = labels / (scan.stem + ".label") if labels else None
            if require_labels and (label is None or not label.exists()):
                continue
            pairs.append((scan, label))
    return pairs


def voxelize(coords_int):
    """Unique voxels of an (N, 3) int array. Returns (unique coords, index of one point per voxel,
    inverse map point -> voxel)."""
    c = coords_int - coords_int.min(0)
    dims = c.max(0) + 1
    key = (c[:, 0].astype(np.int64) * dims[1] + c[:, 1]) * dims[2] + c[:, 2]
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    return c[first], first, inverse


class SemanticKittiVoxels:
    """Point clouds -> sparse voxels for the sparse U-Net.

    Training: one random point per voxel with augmentation. Evaluation: every point keeps an
    index into its voxel so per-voxel predictions can be scattered back to all points.
    """

    def __init__(self, root=DEFAULT_ROOT, seqs=TRAIN_SEQS, voxel_size=0.05, max_range=100.0,
                 train=True, step=1):
        self.files = scan_files(root, seqs, require_labels=True, step=step)
        self.voxel_size = voxel_size
        self.max_range = max_range
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        scan_path, label_path = self.files[i]
        pts = read_scan(scan_path)
        labels, moving, _ = read_label(label_path)
        keep = np.abs(pts[:, :2]).max(1) < self.max_range
        pts, labels = pts[keep], labels[keep]
        xyz = pts[:, :3].copy()

        if self.train:
            a = np.random.uniform(0, 2 * np.pi)
            rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], np.float32)
            xyz = xyz @ rot.T
            if np.random.rand() < 0.5:
                xyz[:, 0] = -xyz[:, 0]
            if np.random.rand() < 0.5:
                xyz[:, 1] = -xyz[:, 1]
            xyz *= np.random.uniform(0.95, 1.05)
            xyz += np.random.normal(0, 0.01, xyz.shape).astype(np.float32)

        coords = np.floor(xyz / self.voxel_size).astype(np.int32)
        if self.train:
            # random representative point per voxel instead of always the same one
            order = np.random.permutation(len(xyz))
            vcoords, first_p, _ = voxelize(coords[order])
            first = order[first_p]
        else:
            vcoords, first, inverse = voxelize(coords)
        feats = np.concatenate([xyz[first], pts[first, 3:4], np.linalg.norm(xyz[first, :2], axis=1, keepdims=True) / 50.0], 1)
        out = {
            "coords": vcoords.astype(np.int32),
            "feats": feats.astype(np.float32),
            "labels": labels[first].astype(np.int64),
        }
        if not self.train:
            out["inverse"] = inverse.astype(np.int64)
            out["point_labels"] = labels.astype(np.int64)
            out["point_xyz"] = pts[:, :3]
        return out


def collate_voxels(batch):
    """Stack samples into spconv indices (batch, x, y, z)."""
    import torch
    coords, feats, labels, inverse, point_labels, point_xyz, offsets = [], [], [], [], [], [], [0]
    for b, s in enumerate(batch):
        n = len(s["coords"])
        coords.append(np.concatenate([np.full((n, 1), b, np.int32), s["coords"]], 1))
        feats.append(s["feats"])
        labels.append(s["labels"])
        if "inverse" in s:
            inverse.append(s["inverse"] + offsets[-1])
            point_labels.append(s["point_labels"])
            point_xyz.append(s["point_xyz"])
        offsets.append(offsets[-1] + n)
    out = {
        "coords": torch.from_numpy(np.concatenate(coords)),
        "feats": torch.from_numpy(np.concatenate(feats)),
        "labels": torch.from_numpy(np.concatenate(labels)),
        "batch_size": len(batch),
    }
    if inverse:
        out["inverse"] = torch.from_numpy(np.concatenate(inverse))
        out["point_labels"] = torch.from_numpy(np.concatenate(point_labels))
        out["point_xyz"] = torch.from_numpy(np.concatenate(point_xyz))
    return out


__all__ = ["IGNORE", "TRAIN_SEQS", "VAL_SEQS", "read_scan", "read_label", "scan_files",
           "voxelize", "SemanticKittiVoxels", "collate_voxels"]


def read_lidar_poses(root, seq):
    """(N, 4, 4) sensor-to-world poses of the lidar for every frame of a sequence.

    KITTI poses.txt gives left-camera poses; calib.txt 'Tr' maps lidar -> camera, so
    T_world_lidar = Tr^-1 @ P_cam @ Tr (the SemanticKITTI convention)."""
    seq_dir = _find_seq_dir(root, seq, "velodyne").parent
    poses_file = seq_dir / "poses.txt"
    if not poses_file.exists():
        labels = _find_seq_dir(root, seq, "labels")
        poses_file = labels.parent / "poses.txt"
    calib_file = seq_dir / "calib.txt"
    if not calib_file.exists():
        for s in _sequence_dirs(str(root)):
            if (s / seq / "calib.txt").exists():
                calib_file = s / seq / "calib.txt"
    tr = np.eye(4)
    for line in calib_file.read_text().splitlines():
        if line.startswith("Tr:"):
            tr[:3] = np.array(line.split()[1:], dtype=np.float64).reshape(3, 4)
    p = np.loadtxt(poses_file).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (len(p), 1, 1))
    poses[:, :3] = p
    return np.linalg.inv(tr) @ poses @ tr
