import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _fft_abs_shift(x_nhwc: torch.Tensor):

    x = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x32 = x.to(torch.float32)
    F_complex = torch.fft.fft2(x32, norm='ortho')
    F_complex = torch.fft.fftshift(F_complex, dim=(-2, -1))
    A = torch.abs(F_complex).to(torch.float32)
    return A


def _build_lf_mask(style_module, shape, device, dtype):

    B, C, H, W = shape
    min_hw = float(min(H, W))

    r_logits = style_module.r_logits
    r_min = style_module.r_min
    r_max = style_module.r_max

    rc_alpha = getattr(style_module, "rc_alpha", 0.25)
    rc_w_floor_px = getattr(style_module, "rc_w_floor_px", 4.0)

    r_c  = r_min + (r_max - r_min) * torch.sigmoid(r_logits)
    r_pix = (r_c * min_hw).view(1, C, 1, 1)


    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    cy, cx = H // 2, W // 2
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).view(1, 1, H, W)


    w = torch.clamp(rc_alpha * r_pix, min=rc_w_floor_px)


    t = ((dist - r_pix) / w).clamp(0.0, 1.0)
    M = 0.5 * (1.0 + torch.cos(math.pi * t))
    M = M.expand(B, C, H, W).contiguous()
    return M


def _masked_gram_from_amp(A: torch.Tensor, M: torch.Tensor, eps: float = 1e-12):

    B, C, H, W = A.shape


    HW = float(H * W)
    S_pix  = M.sum(dim=(-2, -1), keepdim=True)
    S_frac = (S_pix / HW).clamp_min(eps)


    A_w = A * torch.sqrt(M.clamp_min(eps))
    A_w = A_w / torch.sqrt(S_frac)


    feats = A_w.view(B, C, H * W)
    G = torch.bmm(feats, feats.transpose(1, 2))
    G = G / (C * HW)
    return G


def lowfreq_efficiency_loss(
    x_clean_nhwc: torch.Tensor,
    x_perturbed_nhwc: torch.Tensor,
    style_module,
    weight: float = 1.0,
    **kwargs
) -> torch.Tensor:


    A_clean = _fft_abs_shift(x_clean_nhwc)
    A_pert  = _fft_abs_shift(x_perturbed_nhwc)


    B, C, H, W = A_clean.shape
    M = _build_lf_mask(style_module, (B, C, H, W), A_clean.device, A_clean.dtype)
    M = M.detach()


    G_clean = _masked_gram_from_amp(A_clean, M)
    G_pert  = _masked_gram_from_amp(A_pert,  M)


    diff = G_clean - G_pert
    fro_per = torch.linalg.norm(diff, ord='fro', dim=(1, 2))
    L_eff = - fro_per.mean()

    return L_eff * weight


def dice_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    smooth: float = 1e-6
):
    mask = (targets != ignore_index).unsqueeze(1)
    valid_targets = targets.clone()
    valid_targets[targets == ignore_index] = 0
    targets_one_hot = F.one_hot(valid_targets, num_classes=num_classes)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

    preds = F.softmax(preds, dim=1)
    preds = preds * mask
    targets_one_hot = targets_one_hot * mask

    dims = (0, 2, 3)
    intersection = torch.sum(preds * targets_one_hot, dims)
    cardinality = torch.sum(preds + targets_one_hot, dims)
    dice = (2. * intersection + smooth) / (cardinality + smooth)
    loss = 1 - dice
    return loss.mean()


def compute_loss(
    class_logits: torch.Tensor,
    masks: torch.Tensor,
    loss_fn: nn.Module,
    num_classes: int,
    ignore_index: int = 255,
    aux_classifiers: nn.ModuleList = None,
    aux_features=None,
    aux_weight: float = 0.0,
    loss_weights=[1, 1]
):

    loss_ce = loss_fn(class_logits, masks)
    loss_dice = dice_loss(class_logits, masks, num_classes=num_classes, ignore_index=ignore_index)
    main_loss = loss_weights[0] * loss_ce + loss_weights[1] * loss_dice

    total_aux_loss = torch.tensor(0.0, device=class_logits.device)
    if aux_classifiers is not None and aux_features is not None:
        aux_losses = []
        for aux_feat, aux_cls in zip(aux_features, aux_classifiers):
            aux_logits = aux_cls(aux_feat)
            loss_aux_ce = loss_fn(aux_logits, masks)
            aux_losses.append(loss_aux_ce)
        if aux_losses:
            aux_loss_mean = torch.mean(torch.stack(aux_losses))
            total_aux_loss = aux_loss_mean * aux_weight

    total_loss = main_loss + total_aux_loss
    return total_loss, loss_ce.item(), loss_dice.item(), total_aux_loss.item()


def compute_consistency_loss(
    class_logits_clean: torch.Tensor,
    class_logits_with_style: torch.Tensor,
    weight: float = 1.0,
    T: float = 1.0,
    conf_thresh: float = 0.0,
    detach_teacher: bool = True,
    eps: float = 1e-6,
):

    assert class_logits_clean.shape == class_logits_with_style.shape, "Logit shapes must match."
    N, C, H, W = class_logits_clean.shape


    T = float(max(T, eps))


    if detach_teacher:
        with torch.no_grad():
            teacher_probs = F.softmax((class_logits_clean / T).to(torch.float32), dim=1)
    else:
        teacher_probs = F.softmax((class_logits_clean / T).to(torch.float32), dim=1)

    student_probs = F.softmax((class_logits_with_style / T).to(torch.float32), dim=1)


    diff_sq = (student_probs - teacher_probs) ** 2
    per_pixel_diff = diff_sq.sum(dim=1)


    if conf_thresh > 0.0:
        with torch.no_grad():
            teacher_conf, _ = teacher_probs.max(dim=1)
            conf_mask = (teacher_conf >= conf_thresh).to(per_pixel_diff.dtype)
    else:
        conf_mask = torch.ones_like(per_pixel_diff, dtype=per_pixel_diff.dtype)

    denom = conf_mask.sum().clamp_min(1.0)
    mse_cons = (per_pixel_diff * conf_mask).sum() / denom

    return mse_cons * weight
