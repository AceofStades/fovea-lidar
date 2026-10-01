"""Confusion-matrix based IoU, overall and per distance band."""
import numpy as np

DISTANCE_BINS = (0, 10, 20, 40, 60, 100)


class IoU:
    def __init__(self, num_classes, ignore=255):
        self.n, self.ignore = num_classes, ignore
        self.cm = np.zeros((num_classes, num_classes), np.int64)

    def update(self, pred, gt):
        m = gt != self.ignore
        self.cm += np.bincount(gt[m].astype(np.int64) * self.n + pred[m], minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        tp = np.diag(self.cm)
        denom = self.cm.sum(0) + self.cm.sum(1) - tp
        return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)

    def miou(self):
        return float(np.nanmean(self.iou()))

    def accuracy(self):
        return float(np.diag(self.cm).sum() / max(self.cm.sum(), 1))


class IoUByDistance:
    def __init__(self, num_classes, bins=DISTANCE_BINS, ignore=255):
        self.bins = bins
        self.meters = [IoU(num_classes, ignore) for _ in range(len(bins) - 1)]

    def update(self, pred, gt, xyz):
        r = np.linalg.norm(xyz[:, :2], axis=1)
        b = np.digitize(r, self.bins) - 1
        for i, m in enumerate(self.meters):
            sel = b == i
            if sel.any():
                m.update(pred[sel], gt[sel])

    def summary(self):
        return {f"{lo}-{hi}m": (m.miou(), m.accuracy(), int(m.cm.sum()))
                for (lo, hi), m in zip(zip(self.bins[:-1], self.bins[1:]), self.meters)}
