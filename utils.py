
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_metrics(num_classes: int, ignore_index: int=255):
    return {
        'tp': [0] * num_classes,
        'fp': [0] * num_classes,
        'fn': [0] * num_classes,
        'intersection': [0] * num_classes,
        'union': [0] * num_classes
    }


def accumulate_metrics(preds, targets, global_metrics, num_classes: int, ignore_index: int=255):
    valid = (targets != ignore_index)
    preds = preds[valid]
    targets = targets[valid]

    for cls in range(num_classes):
        preds_cls = (preds == cls).astype(np.uint8)
        targets_cls = (targets == cls).astype(np.uint8)

        tp = np.logical_and(preds_cls == 1, targets_cls == 1).sum()
        fp = np.logical_and(preds_cls == 1, targets_cls == 0).sum()
        fn = np.logical_and(preds_cls == 0, targets_cls == 1).sum()

        intersection = tp
        union = np.logical_or(preds_cls, targets_cls).sum()

        global_metrics['tp'][cls] += tp
        global_metrics['fp'][cls] += fp
        global_metrics['fn'][cls] += fn
        global_metrics['intersection'][cls] += intersection
        global_metrics['union'][cls] += union
