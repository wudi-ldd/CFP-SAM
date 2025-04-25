import torch
import torch.nn.functional as F

__all__ = [
    "dice_loss",
    "gram_matrix",
    "domain_discrepancy_loss",
    "compute_loss",
    "compute_consistency_loss",
]


def dice_loss(preds, targets, num_classes, ignore_index=255, smooth=1e-6):
    mask = (targets != ignore_index).unsqueeze(1)
    targets_one_hot = F.one_hot(
        torch.where(targets == ignore_index, torch.zeros_like(targets), targets),
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).float()
    preds = F.softmax(preds, dim=1)
    preds, targets_one_hot = preds * mask, targets_one_hot * mask
    dims = (0, 2, 3)
    inter = torch.sum(preds * targets_one_hot, dims)
    card = torch.sum(preds + targets_one_hot, dims)
    dice = (2 * inter + smooth) / (card + smooth)
    return 1 - dice.mean()


def gram_matrix(x):
    x = x.permute(0, 3, 1, 2)
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    g = torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)
    return g


def domain_discrepancy_loss(x_clean, x_pert, weight):
    g1, g2 = gram_matrix(x_clean), gram_matrix(x_pert)
    return torch.norm(g1 - g2, p="fro", dim=(1, 2)).mean() * weight


def compute_loss(
    class_logits,
    masks,
    loss_fn,
    num_classes,
    ignore_index=255,
    aux_classifiers=None,
    aux_features=None,
    aux_weight=0.0,
    loss_weights=[1, 1],
):
    ce = loss_fn(class_logits, masks)
    dice = dice_loss(class_logits, masks, num_classes, ignore_index)
    main = loss_weights[0] * ce + loss_weights[1] * dice

    aux_total = torch.tensor(0.0, device=class_logits.device)
    if aux_classifiers and aux_features:
        aux_losses = [loss_fn(cls(feat), masks) for feat, cls in zip(aux_features, aux_classifiers)]
        if aux_losses:
            aux_total = torch.mean(torch.stack(aux_losses)) * aux_weight
    return main + aux_total, ce.item(), dice.item(), aux_total.item()


def compute_consistency_loss(clean_logits, pert_logits, weight=1.0):
    return F.mse_loss(clean_logits, pert_logits, reduction="mean") * weight
