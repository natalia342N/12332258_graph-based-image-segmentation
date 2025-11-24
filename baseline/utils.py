import torch

EPS = 1e-6

def sigmoid_probs(logits):
    return torch.sigmoid(logits)

def binarize_probs(probs, threshold=0.5):
    return (probs > threshold).long()

def compute_confusion(pred_bin, target_bin):
    pred = pred_bin.bool()
    target = target_bin.bool()

    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()
    return tp, fp, fn, tn

def update_metrics(logits, targets, stats):
    probs = sigmoid_probs(logits)
    preds = binarize_probs(probs)

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    tp, fp, fn, tn = compute_confusion(preds, targets)
    stats["tp"] += tp
    stats["fp"] += fp
    stats["fn"] += fn
    stats["tn"] += tn
    stats["total"] += targets.numel()

def compute_epoch_metrics(stats):
    tp = stats["tp"]; fp = stats["fp"]; fn = stats["fn"]; tn = stats["tn"]; total = stats["total"]

    iou_fg = tp / (tp + fp + fn + EPS)

    # Background IoU
    iou_bg = tn / (tn + fn + fp + EPS)

    miou = 0.5 * (iou_fg + iou_bg)
    dice_fg = 2 * tp / (2 * tp + fp + fn + EPS)
    acc = (tp + tn) / (total + EPS)

    return dict(
        iou_fg=iou_fg,
        iou_bg=iou_bg,
        miou=miou,
        dice_fg=dice_fg,
        pixel_acc=acc,
    )
