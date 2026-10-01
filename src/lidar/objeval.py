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
        """Match detections to labelled objects with one joint histogram over (object, box) point
        pairs, so a frame costs O(points) instead of set intersections per pair."""
        cat = np.where(gt_train < NUM_CLASSES, TRAIN_TO_CATEGORY[np.minimum(gt_train, NUM_CLASSES - 1)], 0)
        dyn = ((cat == VEHICLE) | (cat == PEDESTRIAN)) & (gt_instance > 0)
        key = gt_instance.astype(np.int64) * 8 + cat
        uniq, gt_id = np.unique(np.where(dyn, key, -1), return_inverse=True)
        gt_id = gt_id.reshape(-1)
        if uniq[0] == -1:
            uniq = uniq[1:]
        else:
            gt_id = gt_id + 1                       # no background present; keep 0 free for it
        box_id = np.zeros(len(xyz), np.int64)
        for j, idx in enumerate(box_indices):
            box_id[idx] = j + 1
        G, B = len(uniq), len(boxes)
        joint = np.bincount(gt_id * (B + 1) + box_id, minlength=(G + 1) * (B + 1)).reshape(G + 1, B + 1)
        size_g, size_b = joint.sum(1), joint.sum(0)
        matched = set()
        for g in range(1, G + 1):
            if size_g[g] < MIN_GT_POINTS:
                continue
            c = int(uniq[g - 1] % 8)
            k = 0 if c == VEHICLE else 1
            pts = np.flatnonzero(gt_id == g)
            b = self._bin(float(np.hypot(*xyz[pts, :2].mean(0))))
            self.gt[k, b] += 1
            inter = joint[g, 1:]
            iou = inter / np.maximum(size_g[g] + size_b[1:] - inter, 1)
            for j in np.argsort(-iou):
                if iou[j] < self.iou:
                    break
                if j in matched or boxes[j]["category"] != c:
                    continue
                self.tp_gt[k, b] += 1
                self.tp_pred[k, self._bin(boxes[j]["distance"])] += 1
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
