"""Object-level detection metrics by distance, against SemanticKITTI instance labels.

A labelled object is a (class group, instance id) set of points; a detection matches it when the
point-set IoU is at least 0.5 and the map category (vehicle / pedestrian) agrees.
"""
import numpy as np

from .labels import NUM_CLASSES, PEDESTRIAN, TRAIN_TO_CATEGORY, VEHICLE

BINS = (0, 10, 20, 40, 60, 100)
MIN_GT_POINTS = 10


class ObjectEval:
    def __init__(self, bins=BINS, iou=0.5):
        self.bins, self.iou = bins, iou
        n = len(bins) - 1
        self.gt = np.zeros((2, n), int)
        self.tp_gt = np.zeros((2, n), int)    # matches binned by the labelled object's distance (recall)
        self.tp_pred = np.zeros((2, n), int)  # matches binned by the detection's distance (precision)
        self.pred = np.zeros((2, n), int)

    def _bin(self, d):
        return int(np.clip(np.digitize(d, self.bins) - 1, 0, len(self.bins) - 2))

    def update(self, xyz, gt_train, gt_instance, boxes, box_indices):
        cat = np.where(gt_train < NUM_CLASSES, TRAIN_TO_CATEGORY[np.minimum(gt_train, NUM_CLASSES - 1)], 0)
        objects = []
        for c in (VEHICLE, PEDESTRIAN):
            sel = (cat == c) & (gt_instance > 0)
            for inst in np.unique(gt_instance[sel]):
                idx = np.flatnonzero(sel & (gt_instance == inst))
                if len(idx) >= MIN_GT_POINTS:
                    objects.append((c, idx, float(np.hypot(*xyz[idx, :2].mean(0)))))
        matched = set()
        for c, idx, d in objects:
            k = 0 if c == VEHICLE else 1
            b = self._bin(d)
            self.gt[k, b] += 1
            gset = set(idx.tolist())
            for j, (box, pidx) in enumerate(zip(boxes, box_indices)):
                if j in matched or box["category"] != c:
                    continue
                inter = len(gset.intersection(pidx.tolist()))
                if inter and inter / (len(gset) + len(pidx) - inter) >= self.iou:
                    self.tp_gt[k, b] += 1
                    self.tp_pred[k, self._bin(box["distance"])] += 1
                    matched.add(j)
                    break
        for box in boxes:
            k = 0 if box["category"] == VEHICLE else 1
            self.pred[k, self._bin(box["distance"])] += 1

    def summary(self):
        out = {}
        for k, name in enumerate(("vehicle", "pedestrian")):
            rows = {}
            for b, (lo, hi) in enumerate(zip(self.bins[:-1], self.bins[1:])):
                g, p = self.gt[k, b], self.pred[k, b]
                rows[f"{lo}-{hi}m"] = {"recall": self.tp_gt[k, b] / g if g else None,
                                       "precision": self.tp_pred[k, b] / p if p else None, "objects": int(g)}
            g, t, p = self.gt[k].sum(), self.tp_gt[k].sum(), self.pred[k].sum()
            out[name] = {"recall": t / g if g else None, "precision": t / p if p else None, "objects": int(g), "by_distance": rows}
        return out
