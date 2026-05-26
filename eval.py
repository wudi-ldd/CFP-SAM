import os
import cv2
import argparse
import logging
import numpy as np
from tqdm import tqdm
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CONFIG
from datasets import SegmentationDataset, read_split_files
from models.hooks import forward_all, forward_inter
from models.heads import SegmentationHead
from utils import initialize_metrics, accumulate_metrics

from sam2.build_sam import build_sam2
from lora2 import LoRA_sam2

try:
    import ttach as tta
    HAS_TTA = True
except Exception:
    HAS_TTA = False


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _SegModelForTTA(nn.Module):
    def __init__(self, lora_sam_model, seg_head, use_amp: bool = False):
        super().__init__()
        self.lora = lora_sam_model
        self.head = seg_head
        self.use_amp = bool(use_amp)

    def forward(self, images: torch.Tensor):
        if self.use_amp and images.is_cuda:
            with torch.cuda.amp.autocast():
                out_enc = self.lora.sam_model.image_encoder.forward_all(
                    images, style_intervener=None
                )
                logits = self.head(out_enc["backbone_fpn"])
        else:
            out_enc = self.lora.sam_model.image_encoder.forward_all(
                images, style_intervener=None
            )
            logits = self.head(out_enc["backbone_fpn"])

        height, width = images.shape[-2:]
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_tta_wrapper(model: nn.Module, tta_mode: str):
    if (not HAS_TTA) or (tta_mode is None):
        if tta_mode is not None and not HAS_TTA:
            print("[Eval][Warn] ttach is not installed. Running evaluation without TTA.")
        return model

    if tta_mode == "lr":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
        ])
    elif tta_mode == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
            tta.Scale(
                scales=[0.5, 0.75, 1.0, 1.25, 1.5],
                interpolation="bicubic",
                align_corners=False,
            ),
        ])
    else:
        return model

    return tta.SegmentationTTAWrapper(model, transforms, merge_mode="mean")


def _find_latest_run_dir(base_dir: str, dataset_name: str):
    if not os.path.isdir(base_dir):
        return None

    prefix = f"{dataset_name}_"
    candidates = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if os.path.isdir(full) and name.startswith(prefix):
            candidates.append(full)

    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return os.path.basename(candidates[0])


def _resolve_run_name(args_run_name: str, dataset_name: str, base_dir: str):
    if args_run_name is not None and len(args_run_name.strip()) > 0:
        return args_run_name.strip()

    latest = _find_latest_run_dir(base_dir=base_dir, dataset_name=dataset_name)
    if latest is not None:
        return latest

    return dataset_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentation evaluation for CFP-SAM (SAM2+LoRA+Head)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Specify a training run directory name, e.g. beijing_20251212_203830.",
    )
    parser.add_argument(
        "--tta_mode",
        type=str,
        default=None,
        choices=["lr", "d4", "none", "None"],
        help="Override tta_mode in config.py: lr/d4/none.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Override config.py and enable AMP during evaluation.",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Override config.py and disable visualization export.",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="Override config.py and specify the evaluation split file.",
    )
    return parser.parse_args()


def _setup_logger(logs_dir: str, run_name: str):
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger("eval_all")
    for handler in list(logger.handlers):
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    metrics_log_path = os.path.join(logs_dir, "eval_metrics.log")

    file_handler = logging.FileHandler(metrics_log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info(f"===== Eval Start | run_name={run_name} =====")
    return logger, metrics_log_path


def build_pred_color_map(num_classes: int):
    tgrs_rgb_map = {
        0: (255, 255, 255),
        1: (0, 0, 255),
        2: (0, 255, 255),
        3: (0, 255, 0),
        4: (255, 255, 0),
        5: (255, 0, 0),
    }

    extra_rgb_colors = [
        (255, 0, 255),
        (128, 128, 255),
        (255, 128, 0),
        (128, 255, 128),
        (0, 128, 255),
        (255, 128, 128),
        (128, 255, 255),
        (255, 0, 128),
        (128, 0, 255),
        (0, 255, 128),
    ]

    color_map = {}
    for cls in range(num_classes):
        if cls in tgrs_rgb_map:
            color_map[cls] = tgrs_rgb_map[cls]
        else:
            color_map[cls] = extra_rgb_colors[(cls - 6) % len(extra_rgb_colors)]

    return color_map


def colorize_mask(mask: np.ndarray, color_map: dict, ignore_index: int = 255):
    height, width = mask.shape
    color = np.zeros((height, width, 3), dtype=np.uint8)

    for cls, rgb in color_map.items():
        color[mask == cls] = rgb

    color[mask == ignore_index] = (0, 0, 0)
    return color


def main():
    args = parse_args()
    seed_everything(42)

    device = CONFIG["device"]
    dataset_name = CONFIG["dataset_name"]
    test_list = args.test_list if args.test_list is not None else CONFIG.get("val_split", "test.txt")

    num_classes = int(CONFIG["num_classes"])
    ignore_index = int(CONFIG.get("ignore_index", 255))

    eval_classes_cfg = CONFIG.get("eval_classes", None)
    eval_classes = list(range(num_classes)) if eval_classes_cfg is None else list(eval_classes_cfg)

    tta_mode = CONFIG.get("tta_mode", "lr")
    if args.tta_mode is not None:
        tta_mode = None if args.tta_mode in ["none", "None"] else args.tta_mode

    use_amp = bool(CONFIG.get("use_amp_eval", False))
    if args.use_amp:
        use_amp = True

    save_overlay = bool(CONFIG.get("save_overlay", True))
    if args.no_overlay:
        save_overlay = False

    overlay_mode = CONFIG.get("overlay_mode", "save_separate_masks")
    overlay_alpha = float(CONFIG.get("overlay_alpha", 0.2))
    overlay_dir = CONFIG.get("overlay_dir", "predictions")

    eval_batch_size = int(CONFIG.get("eval_batch_size", CONFIG.get("batch_size", 1)))
    eval_num_workers = int(CONFIG.get("eval_num_workers", 8))

    base_dir = CONFIG.get("base_dir", "logs")
    run_name = _resolve_run_name(args.run_name, dataset_name=dataset_name, base_dir=base_dir)

    base_log_dir = os.path.join(base_dir, run_name)
    logs_dir = os.path.join(base_log_dir, "logs")
    weights_dir = os.path.join(base_log_dir, "weights")
    predictions_dir = os.path.join(base_log_dir, overlay_dir)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    logger, metrics_log_path = _setup_logger(logs_dir, run_name)

    logger.info(f"[Eval] dataset_name={dataset_name}")
    logger.info(f"[Eval] run_name={run_name}")
    logger.info(f"[Eval] weights_dir={weights_dir}")
    logger.info(f"[Eval] TEST_LIST={test_list}")
    logger.info(f"[Eval] Classes={eval_classes}, ignore_index={ignore_index}")
    logger.info(f"[Eval] TTA={tta_mode}, AMP={use_amp}, SAVE_OVERLAY={save_overlay}")
    logger.info(f"[Eval] overlay_mode={overlay_mode}, overlay_alpha={overlay_alpha}")
    logger.info(f"[Eval] batch={eval_batch_size}, workers={eval_num_workers}")
    logger.info(f"[Eval] log_file={metrics_log_path}")

    pred_color_map = build_pred_color_map(num_classes)
    logger.info(f"[Eval] TGRS color map (RGB) = {pred_color_map}")

    checkpoint = CONFIG["checkpoint"]
    model_cfg = CONFIG["model_cfg"]
    sam_model = build_sam2(model_cfg, checkpoint, device)

    sam_model.image_encoder.forward_all = MethodType(forward_all, sam_model.image_encoder)
    sam_model.image_encoder.trunk.forward_inter = MethodType(forward_inter, sam_model.image_encoder.trunk)

    for comp in [
        "sam_mask_decoder",
        "sam_prompt_encoder",
        "memory_encoder",
        "memory_attention",
        "mask_downsample",
        "obj_ptr_tpos_proj",
        "obj_ptr_proj",
    ]:
        if hasattr(sam_model, comp):
            delattr(sam_model, comp)

    lora_sam_model = LoRA_sam2(
        sam_model,
        rank=CONFIG["lora_rank"],
        alpha=CONFIG["lora_alpha"],
    ).to(device)
    lora_sam_model.eval()

    use_ldp_effective = bool(CONFIG.get("use_ldp", False)) and (
        len(CONFIG.get("ldp_stage_ids", [])) > 0
    )
    lora_suffix = "_lora_LDP" if use_ldp_effective else "_lora"

    lora_path = os.path.join(weights_dir, f"{CONFIG['save_prefix']}{lora_suffix}.safetensors")
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA parameter file not found: {lora_path}")

    lora_sam_model.load_lora_parameters(lora_path)
    logger.info(f"[Eval] Loaded LoRA: {lora_path}")

    seg_head_path = os.path.join(weights_dir, f"{CONFIG['save_prefix']}{lora_suffix}.pth")
    model_seg_head = SegmentationHead(
        fpn_channels=[256, 256, 256, 256],
        out_channels=num_classes,
        align_corners=False,
    ).to(device)

    if not os.path.exists(seg_head_path):
        raise FileNotFoundError(f"SegmentationHead checkpoint not found: {seg_head_path}")

    model_seg_head.load_state_dict(torch.load(seg_head_path, map_location=device))
    model_seg_head.eval()
    logger.info(f"[Eval] Loaded SegHead: {seg_head_path}")

    base_model = _SegModelForTTA(lora_sam_model, model_seg_head, use_amp=use_amp).to(device).eval()
    model_for_eval = build_tta_wrapper(base_model, tta_mode)
    model_for_eval.eval()

    image_dir = os.path.join("datasets", dataset_name, "images")
    mask_dir = os.path.join("datasets", dataset_name, "masks")
    split_path = os.path.join("datasets", dataset_name, test_list)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Evaluation split file not found: {split_path}")

    test_files = read_split_files(split_path)
    test_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        file_list=test_files,
        mask_size=CONFIG["image_size"],
        is_train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"[Eval] image_dir={image_dir}")
    logger.info(f"[Eval] mask_dir={mask_dir}")
    logger.info(f"[Eval] num_test_images={len(test_dataset)}")
    logger.info(f"[Eval] predictions_dir={predictions_dir}")

    global_metrics = initialize_metrics(num_classes, ignore_index=ignore_index)
    correct_pixels = 0
    total_pixels = 0

    target_h, target_w = CONFIG["image_size"]

    with torch.inference_mode():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()

            logits = model_for_eval(images)
            preds = torch.argmax(logits, dim=1).long()

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for batch_item in range(preds_np.shape[0]):
                pred = preds_np[batch_item]
                mask = masks_np[batch_item]
                accumulate_metrics(
                    pred,
                    mask,
                    global_metrics,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )

                valid = mask != ignore_index
                correct_pixels += int((pred[valid] == mask[valid]).sum())
                total_pixels += int(valid.sum())

            if not save_overlay:
                continue

            start = batch_idx * test_loader.batch_size
            end = start + preds_np.shape[0]

            for batch_item, file_idx in enumerate(range(start, end)):
                if file_idx >= len(test_dataset.image_files):
                    break

                image_file = test_dataset.image_files[file_idx]
                img_path = os.path.join(image_dir, image_file)

                if overlay_mode == "save_overlaid_images":
                    orig = cv2.imread(img_path)
                    if orig is None:
                        logger.error(f"Failed to read image: {img_path}")
                        continue

                    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                    orig = cv2.resize(orig, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    pred_color = colorize_mask(
                        preds_np[batch_item],
                        pred_color_map,
                        ignore_index=ignore_index,
                    )
                    gt_color = colorize_mask(
                        masks_np[batch_item],
                        pred_color_map,
                        ignore_index=ignore_index,
                    )

                    overlay_pred = cv2.addWeighted(orig, 1 - overlay_alpha, pred_color, overlay_alpha, 0)
                    overlay_gt = cv2.addWeighted(orig, 1 - overlay_alpha, gt_color, overlay_alpha, 0)

                    pred_save = os.path.join(
                        predictions_dir,
                        os.path.splitext(image_file)[0] + "_pred_overlay.png",
                    )
                    gt_save = os.path.join(
                        predictions_dir,
                        os.path.splitext(image_file)[0] + "_gt_overlay.png",
                    )

                    cv2.imwrite(pred_save, cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(gt_save, cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))
                elif overlay_mode == "save_separate_masks":
                    pred_color = colorize_mask(
                        preds_np[batch_item],
                        pred_color_map,
                        ignore_index=ignore_index,
                    )
                    gt_color = colorize_mask(
                        masks_np[batch_item],
                        pred_color_map,
                        ignore_index=ignore_index,
                    )

                    pred_save = os.path.join(
                        predictions_dir,
                        os.path.splitext(image_file)[0] + "_pred_mask.png",
                    )
                    gt_save = os.path.join(
                        predictions_dir,
                        os.path.splitext(image_file)[0] + "_gt_mask.png",
                    )

                    cv2.imwrite(pred_save, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(gt_save, cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR))
                else:
                    raise ValueError(
                        "overlay_mode must be 'save_overlaid_images' or 'save_separate_masks'"
                    )

    for cls in range(num_classes):
        intersection = global_metrics["intersection"][cls]
        union = global_metrics["union"][cls]
        tp = global_metrics["tp"][cls]
        fp = global_metrics["fp"][cls]
        fn = global_metrics["fn"][cls]

        iou = intersection / (union + 1e-6) if union > 0 else 0.0
        precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall + 1e-6)
            if (precision + recall) > 0
            else 0.0
        )

        msg = (
            f"Class {cls} - IoU: {iou:.4f}, Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f}"
        )
        print(msg)
        logger.info(msg)

    valid_iou = []
    valid_precision = []
    valid_recall = []
    valid_f1 = []

    for cls in eval_classes:
        intersection = global_metrics["intersection"][cls]
        union = global_metrics["union"][cls]
        tp = global_metrics["tp"][cls]
        fp = global_metrics["fp"][cls]
        fn = global_metrics["fn"][cls]

        if union > 0:
            valid_iou.append(intersection / (union + 1e-6))
        if (tp + fp) > 0:
            valid_precision.append(tp / (tp + fp + 1e-6))
        if (tp + fn) > 0:
            valid_recall.append(tp / (tp + fn + 1e-6))
        if (tp + fp + fn) > 0:
            precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall + 1e-6)
                if (precision + recall) > 0
                else 0.0
            )
            valid_f1.append(f1)

    avg_iou = float(np.mean(valid_iou)) if valid_iou else 0.0
    avg_precision = float(np.mean(valid_precision)) if valid_precision else 0.0
    avg_recall = float(np.mean(valid_recall)) if valid_recall else 0.0
    avg_f1 = float(np.mean(valid_f1)) if valid_f1 else 0.0
    oa = correct_pixels / (total_pixels + 1e-6) if total_pixels > 0 else 0.0

    summary = (
        f"[Classes={eval_classes}] Average metrics - IoU: {avg_iou:.4f}, "
        f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, "
        f"F1: {avg_f1:.4f}, OA: {oa:.4f}"
        + ("" if tta_mode is None else f"  (TTA={tta_mode}, AMP={use_amp})")
    )
    print(summary)
    logger.info(summary)
    logger.info("===== Eval End =====")


if __name__ == "__main__":
    main()
