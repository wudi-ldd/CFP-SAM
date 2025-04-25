import numpy as np

__all__ = ["initialize_metrics", "accumulate_metrics"]


def initialize_metrics(num_classes, ignore_index=255):
    return {
        "tp": [0] * num_classes,
        "fp": [0] * num_classes,
        "fn": [0] * num_classes,
        "intersection": [0] * num_classes,
        "union": [0] * num_classes,
    }


def accumulate_metrics(preds, targets, global_metrics, num_classes, ignore_index=255):
    valid = targets != ignore_index
    preds, targets = preds[valid], targets[valid]
    for cls in range(num_classes):
        p_cls = preds == cls
        t_cls = targets == cls
        tp = np.logical_and(p_cls, t_cls).sum()
        fp = np.logical_and(p_cls, ~t_cls).sum()
        fn = np.logical_and(~p_cls, t_cls).sum()
        inter = tp
        uni = np.logical_or(p_cls, t_cls).sum()
        global_metrics["tp"][cls] += tp
        global_metrics["fp"][cls] += fp
        global_metrics["fn"][cls] += fn
        global_metrics["intersection"][cls] += inter
        global_metrics["union"][cls] += uni
