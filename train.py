import os
import math
import logging
import numpy as np
from tqdm import tqdm
from types import MethodType
from contextlib import contextmanager
from datetime import datetime
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from sam2.build_sam import build_sam2
from lora2 import LoRA_sam2

from config import CONFIG, SEED
from utils import set_seed, initialize_metrics, accumulate_metrics
from datasets import SegmentationDataset, read_split_files
from models.hooks import forward_all, forward_inter
from models.ldp import MultiStageLDP
from models.heads import SegmentationHead, AuxiliaryClassifier
from losses import compute_loss, compute_consistency_loss, lowfreq_efficiency_loss


@contextmanager
def freeze(module: nn.Module):
    if module is None:
        yield
        return
    flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(module.parameters(), flags):
            p.requires_grad_(f)


@contextmanager
def force_exactly_one_ldp(style_intervener, prefer_indices=None):
    if style_intervener is None:
        yield
        return

    active = list(getattr(style_intervener, "active_indices", []))
    if prefer_indices is not None:
        active = [i for i in active if i in prefer_indices]
    if len(active) == 0:
        yield
        return

    idx = active[torch.randint(low=0, high=len(active), size=(1,)).item()]

    mods, old_ps = [], []
    for i in active:
        m = style_intervener.module_for(i)
        if m is not None and hasattr(m, "p"):
            mods.append(m)
            old_ps.append(m.p)
            m.p = 0.0

    chosen = style_intervener.module_for(idx)
    if chosen is not None and hasattr(chosen, "p"):
        chosen.p = 1.0

    try:
        yield
    finally:
        for m, p in zip(mods, old_ps):
            m.p = p


def main():
    set_seed(SEED)
    device = CONFIG['device']

    checkpoint = CONFIG['checkpoint']
    model_cfg = CONFIG['model_cfg']
    sam_model = build_sam2(model_cfg, checkpoint, device)

    sam_model.image_encoder.forward_all = MethodType(forward_all, sam_model.image_encoder)
    sam_model.image_encoder.trunk.forward_inter = MethodType(forward_inter, sam_model.image_encoder.trunk)

    sam_model.image_encoder.trunk.use_checkpoint = CONFIG.get("use_checkpoint", True)
    sam_model.image_encoder.trunk.checkpoint_use_reentrant = CONFIG.get("checkpoint_use_reentrant", False)
    sam_model.image_encoder.trunk.checkpoint_layers = CONFIG.get("checkpoint_layers", None)

    for comp in [
        'sam_mask_decoder', 'sam_prompt_encoder', 'memory_encoder', 'memory_attention',
        'mask_downsample', 'obj_ptr_tpos_proj', 'obj_ptr_proj'
    ]:
        if hasattr(sam_model, comp):
            delattr(sam_model, comp)

    lora_sam_model = LoRA_sam2(
        sam_model,
        rank=CONFIG['lora_rank'],
        alpha=CONFIG['lora_alpha']
    ).to(device)

    image_dir = os.path.join('datasets', CONFIG['dataset_name'], 'images')
    mask_dir = os.path.join('datasets', CONFIG['dataset_name'], 'masks')

    train_split = CONFIG.get('train_split', 'train.txt')
    val_split = CONFIG.get('val_split', 'test.txt')

    train_files = read_split_files(f'datasets/{CONFIG["dataset_name"]}/{train_split}')
    val_files = read_split_files(f'datasets/{CONFIG["dataset_name"]}/{val_split}')

    train_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_list=train_files,
        mask_size=CONFIG['image_size'],
        is_train=True
    )
    val_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_list=val_files,
        mask_size=CONFIG['image_size'],
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    for i, (images, masks) in enumerate(train_loader):
        print(f'Train Batch {i}:')
        print(f'Images shape: {images.shape}')
        print(f'Masks shape: {masks.shape}')
        print(f'Mask unique values: {torch.unique(masks)}')
        break

    for i, (images, masks) in enumerate(val_loader):
        print(f'Val Batch {i}:')
        print(f'Images shape: {images.shape}')
        print(f'Masks shape: {masks.shape}')
        print(f'Mask unique values: {torch.unique(masks)}')
        break

    model_seg_head = SegmentationHead(
        fpn_channels=[256, 256, 256, 256],
        out_channels=CONFIG['num_classes'],
        align_corners=False
    ).to(device)

    aux_classifiers = nn.ModuleList([
        AuxiliaryClassifier(in_channels=144,  num_classes=CONFIG['num_classes'], output_size=CONFIG['image_size']).to(device),
        AuxiliaryClassifier(in_channels=288,  num_classes=CONFIG['num_classes'], output_size=CONFIG['image_size']).to(device),
        AuxiliaryClassifier(in_channels=576,  num_classes=CONFIG['num_classes'], output_size=CONFIG['image_size']).to(device),
        AuxiliaryClassifier(in_channels=1152, num_classes=CONFIG['num_classes'], output_size=CONFIG['image_size']).to(device),
    ]).to(device)

    for p in model_seg_head.parameters():
        p.requires_grad = True
    for aux in aux_classifiers:
        for p in aux.parameters():
            p.requires_grad = True

    for p in lora_sam_model.sam_model.image_encoder.parameters():
        p.requires_grad = False
    for layer in lora_sam_model.A_weights_q + lora_sam_model.B_weights_q:
        for p in layer.parameters():
            p.requires_grad = True
    for layer in lora_sam_model.A_weights_v + lora_sam_model.B_weights_v:
        for p in layer.parameters():
            p.requires_grad = True

    style_intervener = None
    optimizer_style = None
    scheduler_style = None

    if CONFIG['use_ldp']:
        stage_ids = CONFIG.get('ldp_stage_ids', [0, 1, 2, 3])
        ldp_p = CONFIG.get('ldp_p', 0.5)
        style_intervener = MultiStageLDP(stage_ids=stage_ids, p=ldp_p).to(device)

        optimizer_style = torch.optim.AdamW(
            style_intervener.parameters(),
            lr=CONFIG['learning_rate_style'],
            betas=CONFIG['betas'],
            weight_decay=CONFIG['weight_decay']
        )

    backbone_trainable_params = list(filter(lambda p: p.requires_grad, lora_sam_model.parameters()))
    head_trainable_params = list(model_seg_head.parameters()) + list(aux_classifiers.parameters())

    optimizer_backbone = torch.optim.AdamW(
        backbone_trainable_params,
        lr=CONFIG['learning_rate_backbone'],
        betas=CONFIG['betas'],
        weight_decay=CONFIG['weight_decay']
    )
    optimizer_head = torch.optim.AdamW(
        head_trainable_params,
        lr=CONFIG['learning_rate_head'],
        betas=CONFIG['betas'],
        weight_decay=CONFIG['weight_decay']
    )

    scaler_main = GradScaler()
    scaler_style = GradScaler()

    loss_fn = nn.CrossEntropyLoss(ignore_index=CONFIG['ignore_index'])

    num_epochs = CONFIG['num_epochs']
    warmup_epochs = 3
    min_lr_factor = 0.01

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float((epoch + 1) / warmup_epochs)
        else:
            cosine_decay = 0.5 * (1 + math.cos((epoch - warmup_epochs) * math.pi / (num_epochs - warmup_epochs)))
            return float(min_lr_factor + (1 - min_lr_factor) * cosine_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(optimizer_backbone, lr_lambda=lr_lambda)
    scheduler_head = torch.optim.lr_scheduler.LambdaLR(optimizer_head, lr_lambda=lr_lambda)
    if optimizer_style is not None:
        scheduler_style = torch.optim.lr_scheduler.LambdaLR(optimizer_style, lr_lambda=lr_lambda)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{CONFIG['dataset_name']}_{run_time}"

    base_log_dir = os.path.join(CONFIG['base_dir'], run_name)
    logs_dir = os.path.join(base_log_dir, 'logs')
    weights_dir = os.path.join(base_log_dir, 'weights')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    config_snapshot_path = os.path.join(base_log_dir, "config_snapshot.py")
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        f.write(f"SEED = {SEED}\n\n")
        f.write("CONFIG = ")
        f.write(pprint.pformat(CONFIG, width=120, sort_dicts=False))
        f.write("\n")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(logs_dir, CONFIG['log_file']),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    best_iou = float('-inf')
    best_epoch = 0

    num_classes = CONFIG['num_classes']
    AUX_WEIGHT = CONFIG['aux_weight']
    CONSISTENCY_WEIGHT = CONFIG['consistency_weight']
    DOMAIN_DISCREPANCY_WEIGHT = CONFIG['domain_discrepancy_weight']
    ignore_index = CONFIG['ignore_index']

    for epoch in range(num_epochs):
        lora_sam_model.train()
        model_seg_head.train()
        aux_classifiers.train()
        if style_intervener is not None:
            style_intervener.train()

        total_min_loss = 0.0
        total_max_loss = 0.0
        total_ce_loss = 0.0
        total_dice_loss = 0.0
        total_aux_loss = 0.0
        total_consistency_loss = 0.0
        total_discrepancy_loss = 0.0
        num_batches = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()

            if images.dim() != 4 or masks.dim() != 3:
                logging.error(f"Invalid input shape: images {images.shape}, masks {masks.shape}")
                continue

            unique_values = torch.unique(masks)
            if not torch.all((unique_values < num_classes) | (unique_values == ignore_index)):
                logging.error(f"Mask contains invalid class values: {unique_values}")
                continue

            if style_intervener is not None and len(getattr(style_intervener, "active_indices", [])) > 0:
                optimizer_style.zero_grad(set_to_none=True)

                with torch.no_grad():
                    with autocast():
                        out_clean_max = lora_sam_model.sam_model.image_encoder.forward_all(
                            images, style_intervener=None
                        )

                with freeze(lora_sam_model):
                    with autocast():
                        out_pert_max = lora_sam_model.sam_model.image_encoder.forward_all(
                            images, style_intervener=style_intervener
                        )

                active_ids = list(style_intervener.active_indices)
                L_eff_total = None
                if len(active_ids) > 0:
                    L_sum = 0.0
                    for idx in active_ids:
                        mod = style_intervener.module_for(idx)
                        if mod is None:
                            continue
                        x_clean = out_clean_max['stages_nhwc'][idx]
                        x_pert = out_pert_max['stages_nhwc'][idx]
                        L_eff_i = lowfreq_efficiency_loss(
                            x_clean, x_pert, style_module=mod, weight=DOMAIN_DISCREPANCY_WEIGHT
                        )
                        L_sum = L_sum + L_eff_i
                    if isinstance(L_sum, torch.Tensor):
                        L_eff_total = L_sum / float(len(active_ids)) if len(active_ids) > 0 else None

                if (L_eff_total is None) or (not L_eff_total.requires_grad):
                    with freeze(lora_sam_model):
                        with autocast():
                            with force_exactly_one_ldp(style_intervener, prefer_indices=active_ids):
                                out_pert_max = lora_sam_model.sam_model.image_encoder.forward_all(
                                    images, style_intervener=style_intervener
                                )
                    if len(active_ids) > 0:
                        L_sum = 0.0
                        for idx in active_ids:
                            mod = style_intervener.module_for(idx)
                            if mod is None:
                                continue
                            x_clean = out_clean_max['stages_nhwc'][idx]
                            x_pert = out_pert_max['stages_nhwc'][idx]
                            L_eff_i = lowfreq_efficiency_loss(
                                x_clean, x_pert, style_module=mod, weight=DOMAIN_DISCREPANCY_WEIGHT
                            )
                            L_sum = L_sum + L_eff_i
                        if isinstance(L_sum, torch.Tensor):
                            L_eff_total = L_sum / float(len(active_ids)) if len(active_ids) > 0 else None
                        else:
                            L_eff_total = None

                if (L_eff_total is not None) and L_eff_total.requires_grad:
                    scaler_style.scale(L_eff_total).backward()
                    scaler_style.step(optimizer_style)
                    scaler_style.update()

                    total_max_loss += L_eff_total.item()
                    total_discrepancy_loss += (-L_eff_total).item()
                else:
                    total_max_loss += 0.0
                    total_discrepancy_loss += 0.0

                optimizer_backbone.zero_grad(set_to_none=True)
                optimizer_head.zero_grad(set_to_none=True)

                with freeze(style_intervener):
                    with autocast():
                        out_clean_min = lora_sam_model.sam_model.image_encoder.forward_all(
                            images, style_intervener=None
                        )
                        out_pert_min = lora_sam_model.sam_model.image_encoder.forward_all(
                            images, style_intervener=style_intervener
                        )

                        class_logits_clean = model_seg_head(out_clean_min['backbone_fpn'])
                        class_logits_with_style = model_seg_head(out_pert_min['backbone_fpn'])

                        loss_min_clean, loss_ce_min_clean, loss_dice_min_clean, loss_aux_min_clean = compute_loss(
                            class_logits=class_logits_clean,
                            masks=masks,
                            loss_fn=loss_fn,
                            num_classes=num_classes,
                            ignore_index=ignore_index,
                            aux_classifiers=aux_classifiers,
                            aux_features=out_clean_min['trunk_features'],
                            aux_weight=AUX_WEIGHT,
                            loss_weights=[1, 1]
                        )

                        loss_min_perturbed, loss_ce_min_perturbed, loss_dice_min_perturbed, loss_aux_min_perturbed = compute_loss(
                            class_logits=class_logits_with_style,
                            masks=masks,
                            loss_fn=loss_fn,
                            num_classes=num_classes,
                            ignore_index=ignore_index,
                            aux_classifiers=aux_classifiers,
                            aux_features=out_pert_min['trunk_features'],
                            aux_weight=AUX_WEIGHT,
                            loss_weights=[1, 1]
                        )

                        consistency_loss = compute_consistency_loss(
                            class_logits_clean=class_logits_clean,
                            class_logits_with_style=class_logits_with_style,
                            weight=CONSISTENCY_WEIGHT,
                            detach_teacher=True,
                        )

                        loss_total_min = 0.5 * loss_min_clean + 0.5 * loss_min_perturbed + consistency_loss

                scaler_main.scale(loss_total_min).backward()
                scaler_main.step(optimizer_backbone)
                scaler_main.step(optimizer_head)
                scaler_main.update()

                total_min_loss += loss_total_min.item()

                total_ce_loss += 0.5 * (loss_ce_min_clean + loss_ce_min_perturbed)
                total_dice_loss += 0.5 * (loss_dice_min_clean + loss_dice_min_perturbed)
                total_aux_loss += 0.5 * (loss_aux_min_clean + loss_aux_min_perturbed)

                total_consistency_loss += consistency_loss.item()
                num_batches += 1

            else:
                optimizer_backbone.zero_grad(set_to_none=True)
                optimizer_head.zero_grad(set_to_none=True)

                with autocast():
                    out = lora_sam_model.sam_model.image_encoder.forward_all(images, style_intervener=None)
                    class_logits = model_seg_head(out['backbone_fpn'])
                    loss, loss_ce, loss_dice, loss_aux = compute_loss(
                        class_logits=class_logits,
                        masks=masks,
                        loss_fn=loss_fn,
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        aux_classifiers=aux_classifiers,
                        aux_features=out['trunk_features'],
                        aux_weight=AUX_WEIGHT,
                        loss_weights=[1, 1]
                    )

                scaler_main.scale(loss).backward()
                scaler_main.step(optimizer_backbone)
                scaler_main.step(optimizer_head)
                scaler_main.update()

                total_min_loss += loss.item()
                total_ce_loss += loss_ce
                total_dice_loss += loss_dice
                total_aux_loss += loss_aux
                num_batches += 1

        avg_train_loss = total_min_loss / num_batches if num_batches > 0 else 0
        avg_ce_loss = total_ce_loss / num_batches if num_batches > 0 else 0
        avg_dice_loss = total_dice_loss / num_batches if num_batches > 0 else 0
        avg_aux_loss = total_aux_loss / num_batches if num_batches > 0 else 0

        if style_intervener is not None and len(getattr(style_intervener, "active_indices", [])) > 0:
            avg_max_loss = total_max_loss / num_batches if num_batches > 0 else 0
            avg_consistency_loss = total_consistency_loss / num_batches if num_batches > 0 else 0
            avg_discrepancy_loss = total_discrepancy_loss / num_batches if num_batches > 0 else 0
        else:
            avg_max_loss = 0.0
            avg_consistency_loss = 0.0
            avg_discrepancy_loss = 0.0

        lora_sam_model.eval()
        model_seg_head.eval()
        aux_classifiers.eval()
        if style_intervener is not None:
            style_intervener.eval()

        val_loss = 0
        num_val_batches = 0
        global_metrics_val = initialize_metrics(num_classes, ignore_index=ignore_index)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).long()

                if images.dim() != 4 or masks.dim() != 3:
                    logging.error(f"Invalid input shape: images {images.shape}, masks {masks.shape}")
                    continue

                unique_values = torch.unique(masks)
                if not torch.all((unique_values < num_classes) | (unique_values == ignore_index)):
                    logging.error(f"Mask contains invalid class values: {unique_values}")
                    continue

                with autocast():
                    out = lora_sam_model.sam_model.image_encoder.forward_all(images, style_intervener=None)
                    class_logits = model_seg_head(out['backbone_fpn'])
                    loss, loss_ce, loss_dice, _ = compute_loss(
                        class_logits=class_logits,
                        masks=masks,
                        loss_fn=loss_fn,
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        aux_classifiers=None,
                        aux_features=None,
                        aux_weight=0.0,
                        loss_weights=[1, 1]
                    )

                val_loss += loss.item()
                num_val_batches += 1

                preds = torch.argmax(class_logits, dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()
                for pred, mask in zip(preds, masks_np):
                    accumulate_metrics(pred, mask, global_metrics_val, num_classes=num_classes, ignore_index=ignore_index)

        tp = global_metrics_val['tp']
        fp = global_metrics_val['fp']
        fn = global_metrics_val['fn']
        intersection = global_metrics_val['intersection']
        union = global_metrics_val['union']

        iou_per_class = [
            intersection[cls] / (union[cls] + 1e-6)
            for cls in range(num_classes) if union[cls] > 0
        ]
        precision_per_class = [
            tp[cls] / (tp[cls] + fp[cls] + 1e-6)
            for cls in range(num_classes) if (tp[cls] + fp[cls]) > 0
        ]
        recall_per_class = [
            tp[cls] / (tp[cls] + fn[cls] + 1e-6)
            for cls in range(num_classes) if (tp[cls] + fn[cls]) > 0
        ]

        f1_per_class = []
        for cls in range(num_classes):
            p = tp[cls] / (tp[cls] + fp[cls] + 1e-6) if (tp[cls] + fp[cls]) > 0 else 0.0
            r = tp[cls] / (tp[cls] + fn[cls] + 1e-6) if (tp[cls] + fn[cls]) > 0 else 0.0
            f1 = (2 * p * r / (p + r + 1e-6)) if (p + r) > 0 else 0.0
            if (tp[cls] + fp[cls] + fn[cls]) > 0:
                f1_per_class.append(f1)

        avg_iou = np.mean(iou_per_class) if iou_per_class else 0.0
        avg_precision = np.mean(precision_per_class) if precision_per_class else 0.0
        avg_recall = np.mean(recall_per_class) if recall_per_class else 0.0
        avg_f1 = np.mean(f1_per_class) if f1_per_class else 0.0
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0

        current_lr_backbone = optimizer_backbone.param_groups[0]['lr']
        current_lr_head = optimizer_head.param_groups[0]['lr']
        current_lr_style = optimizer_style.param_groups[0]['lr'] if optimizer_style is not None else None

        if style_intervener is not None and len(getattr(style_intervener, "active_indices", [])) > 0:
            log_message = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Avg Train Loss: {avg_train_loss:.4f}, "
                f"Avg Val Loss: {avg_val_loss:.4f}, "
                f"Avg CE Loss: {avg_ce_loss:.4f}, "
                f"Avg Dice Loss: {avg_dice_loss:.4f}, "
                f"Avg Aux Loss: {avg_aux_loss:.4f}, "
                f"Avg Consistency Loss: {avg_consistency_loss:.4f}, "
                f"Avg Discrepancy Loss: {avg_discrepancy_loss:.4f}, "
                f"Avg IoU: {avg_iou:.4f}, "
                f"Avg F1: {avg_f1:.4f}, "
                f"Avg Precision: {avg_precision:.4f}, "
                f"Avg Recall: {avg_recall:.4f}, "
                f"LR Backbone: {current_lr_backbone:.6f}, "
                f"LR Head: {current_lr_head:.6f}, "
                f"LR Style: {current_lr_style:.6f}"
            )
        else:
            log_message = (
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Avg Train Loss: {avg_train_loss:.4f}, "
                f"Avg Val Loss: {avg_val_loss:.4f}, "
                f"Avg CE Loss: {avg_ce_loss:.4f}, "
                f"Avg Dice Loss: {avg_dice_loss:.4f}, "
                f"Avg IoU: {avg_iou:.4f}, "
                f"Avg F1: {avg_f1:.4f}, "
                f"Avg Precision: {avg_precision:.4f}, "
                f"Avg Recall: {avg_recall:.4f}, "
                f"LR Backbone: {current_lr_backbone:.6f}, "
                f"LR Head: {current_lr_head:.6f}"
            )

        logging.info(log_message)
        print(log_message)

        if avg_iou > best_iou:
            best_iou = avg_iou
            best_epoch = epoch + 1

            lora_suffix = "_lora"
            if style_intervener is not None and len(getattr(style_intervener, "active_indices", [])) > 0:
                lora_suffix += "_LDP"

            lora_path = os.path.join(weights_dir, f"{CONFIG['save_prefix']}{lora_suffix}.safetensors")
            checkpoint_path = os.path.join(weights_dir, f"{CONFIG['save_prefix']}{lora_suffix}.pth")

            if hasattr(lora_sam_model, 'save_lora_parameters'):
                lora_sam_model.save_lora_parameters(lora_path)
            else:
                logging.warning("lora_sam_model does not define save_lora_parameters.")

            torch.save(model_seg_head.state_dict(), checkpoint_path)

            save_message = f"Best model (by IoU) saved at epoch {best_epoch} with Avg IoU {best_iou:.4f}"
            logging.info(save_message)
            print(save_message)

        scheduler_backbone.step()
        scheduler_head.step()
        if scheduler_style is not None:
            scheduler_style.step()

    logging.info("Training completed.")
    print("Training completed.")


if __name__ == "__main__":
    main()
