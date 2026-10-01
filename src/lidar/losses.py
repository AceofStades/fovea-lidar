"""Cross-entropy + Lovasz-softmax (Berman et al. 2018), the usual pair for lidar segmentation."""
import torch
import torch.nn.functional as F


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(probs, labels, ignore=255):
    valid = labels != ignore
    probs, labels = probs[valid], labels[valid]
    if labels.numel() == 0:
        return probs.sum() * 0
    losses = []
    for c in torch.unique(labels):
        fg = (labels == c).float()
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    return torch.stack(losses).mean()


def seg_loss(logits, labels, class_weights=None, ignore=255, lovasz_weight=1.0):
    ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore)
    lv = lovasz_softmax(F.softmax(logits.float(), 1), labels, ignore)
    return ce + lovasz_weight * lv, ce.detach(), lv.detach()
