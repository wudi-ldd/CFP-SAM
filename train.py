"""Main training script.

$ python train.py
"""
import os
import math
import logging
import random
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam2.build_sam import build_sam2

from config import CONFIG
from datasets import SegmentationDataset, read_split_files
from heads import SegmentationHead, AuxiliaryClassifier
from losses import (
    compute_consistency_loss,
    compute_loss,
    domain_discrepancy_loss,
)
from metrics import accumulate_metrics, initialize_metrics
from ldp import LearnableDomainPerturbation
from lora2 import LoRA_sam2

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ------------------------------------------------------------
# Monkey-patch SAM-2 encoder
# ------------------------------------------------------------

def _forward_all(self, x: torch.Tensor, style_intervener=None):
    inter_x, trunk_feats = self.trunk.forward_inter(x, style_intervener)
    fpn_feats, pos = self.neck(trunk_feats)
    return {
        "vision_features": fpn_feats[-1],
        "vision_pos_enc": pos,
        "backbone_fpn": fpn_feats,
        "trunk_features": trunk_feats,
        "intervened_features": inter_x,
    }


def _forward_inter(self, x: torch.Tensor, style_intervener=None):
    x = self.patch_embed(x)
    x = x + self._get_pos_embed(x.shape[1:3])
    if style_intervener is not None:
        x = style_intervener(x)
        inter = x.clone()
    else:
        inter = x.clone()
    trunk_feats = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
            trunk_feats.append(x.permute(0, 3, 1, 2))
    return inter, trunk_feats

# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------
device = CONFIG["device"]
sam = build_sam2(CONFIG["model_cfg"], CONFIG["checkpoint"], device)
sam.image_encoder.forward_all = MethodType(_forward_all, sam.image_encoder)
sam.image_encoder.trunk.forward_inter = MethodType(
    _forward_inter, sam.image_encoder.trunk
)
# Remove unused modules
def _del_if(obj, attr):
    if hasattr(obj, attr):
        delattr(obj, attr)
for u in [
    "sam_mask_decoder", "sam_prompt_encoder", "memory_encoder", "memory_attention",
    "mask_downsample", "obj_ptr_tpos_proj", "obj_ptr_proj"
]: _del_if(sam, u)

lora_sam = LoRA_sam2(sam, rank=CONFIG["lora_rank"], alpha=CONFIG["lora_alpha"]).to(device)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
img_dir = os.path.join("datasets", CONFIG["dataset_name"], "images")
msk_dir = os.path.join("datasets", CONFIG["dataset_name"], "masks")
train_ids = read_split_files(f"datasets/{CONFIG['dataset_name']}/train.txt")
val_ids = read_split_files(f"datasets/{CONFIG['dataset_name']}/val.txt")
train_ds = SegmentationDataset(img_dir, msk_dir, train_ids, CONFIG["image_size"], True)
val_ds = SegmentationDataset(img_dir, msk_dir, val_ids, CONFIG["image_size"], False)
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8)

# ------------------------------------------------------------
# Heads & Aux classifiers
# ------------------------------------------------------------
seg_head = SegmentationHead([256, 256, 256, 256], CONFIG["num_classes"], align_corners=False).to(device)
aux_clfs = nn.ModuleList([
    AuxiliaryClassifier(144, CONFIG["num_classes"], CONFIG["image_size"]).to(device),
    AuxiliaryClassifier(288, CONFIG["num_classes"], CONFIG["image_size"]).to(device),
    AuxiliaryClassifier(576, CONFIG["num_classes"], CONFIG["image_size"]).to(device),
    AuxiliaryClassifier(1152, CONFIG["num_classes"], CONFIG["image_size"]).to(device),
])

# ------------------------------------------------------------
# Freeze backbone & unfreeze LoRA, head, aux
# ------------------------------------------------------------
for p in lora_sam.sam_model.image_encoder.parameters(): p.requires_grad = False
for m in list(lora_sam.A_weights_q)+list(lora_sam.B_weights_q)+list(lora_sam.A_weights_v)+list(lora_sam.B_weights_v):
    for p in m.parameters(): p.requires_grad = True
for p in seg_head.parameters(): p.requires_grad = True
for m in aux_clfs:
    for p in m.parameters(): p.requires_grad = True

# Optional LDP
ldp = None
if CONFIG["use_ldp"]:
    ldp = LearnableDomainPerturbation(144, p=1.0).to(device)
    for p in ldp.parameters(): p.requires_grad = True

# ------------------------------------------------------------
# Optimizers & Schedulers
# ------------------------------------------------------------
backbone_opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, lora_sam.parameters()),
    lr=CONFIG["learning_rate_backbone"], betas=CONFIG["betas"], weight_decay=CONFIG["weight_decay"]
)
head_opt = torch.optim.AdamW(
    list(seg_head.parameters())+list(aux_clfs.parameters()),
    lr=CONFIG["learning_rate_head"], betas=CONFIG["betas"], weight_decay=CONFIG["weight_decay"]
)
style_opt = None
if ldp:
    style_opt = torch.optim.AdamW(ldp.parameters(), lr=CONFIG["learning_rate_style"], betas=CONFIG["betas"], weight_decay=CONFIG["weight_decay"])


def lr_lambda(ep):
    warmup, total, minf = 3, CONFIG["num_epochs"], 0.01
    if ep < warmup: return (ep+1)/warmup
    return minf + (1-minf)*0.5*(1+math.cos((ep-warmup)*math.pi/(total-warmup)))

bb_sched = lr_scheduler.LambdaLR(backbone_opt, lr_lambda)
h_sched = lr_scheduler.LambdaLR(head_opt, lr_lambda)
s_sched = lr_scheduler.LambdaLR(style_opt, lr_lambda) if style_opt else None

loss_fn = nn.CrossEntropyLoss(ignore_index=CONFIG["ignore_index"])

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
log_root = os.path.join(CONFIG["base_dir"], CONFIG["dataset_name"])
log_dir, w_dir = os.path.join(log_root,"logs"), os.path.join(log_root,"weights")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(w_dir, exist_ok=True)
for h in logging.root.handlers[:]: logging.root.removeHandler(h)
logging.basicConfig(
    filename=os.path.join(log_dir,CONFIG["log_file"]), level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
best_score = -float("inf")
metric_w = CONFIG["metric_weights"]

for epoch in range(CONFIG["num_epochs"]):
    # Train
    lora_sam.train(); seg_head.train(); aux_clfs.train()
    if ldp: ldp.train()
    sums = {k:0.0 for k in ["loss","ce","dice","aux","cons","disc"]}; sums["batches"]=0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Train]"):
        imgs, masks = imgs.to(device), masks.to(device).long()
        # LDP adversarial max step
        if ldp:
            style_opt.zero_grad()
            out_c = lora_sam.sam_model.image_encoder.forward_all(imgs, None)
            out_p = lora_sam.sam_model.image_encoder.forward_all(imgs, ldp)
            L_dis = domain_discrepancy_loss(out_c['intervened_features'], out_p['intervened_features'], CONFIG['domain_discrepancy_weight'])
            (-L_dis).backward(retain_graph=True); style_opt.step()
        # Min step
        backbone_opt.zero_grad(); head_opt.zero_grad()
        out_c = lora_sam.sam_model.image_encoder.forward_all(imgs, None)
        logits_c = seg_head(out_c['backbone_fpn'])
        loss_c, ce_c, dice_c, aux_c = compute_loss(logits_c, masks, loss_fn, CONFIG['num_classes'], CONFIG['ignore_index'], aux_clfs, out_c['trunk_features'], CONFIG['aux_weight'])
        if ldp:
            out_p = lora_sam.sam_model.image_encoder.forward_all(imgs, ldp)
            logits_p = seg_head(out_p['backbone_fpn'])
            loss_p, ce_p, dice_p, aux_p = compute_loss(logits_p, masks, loss_fn, CONFIG['num_classes'], CONFIG['ignore_index'], aux_clfs, out_p['trunk_features'], CONFIG['aux_weight'])
            cons = compute_consistency_loss(logits_c, logits_p, CONFIG['consistency_weight'])
            total_loss = 0.5*(loss_c+loss_p) + cons
        else:
            total_loss=loss_c; ce_p=dice_p=aux_p=0.0; cons=torch.tensor(0.0); L_dis=torch.tensor(0.0)
        total_loss.backward(); backbone_opt.step(); head_opt.step()
        # Accumulate
        sums['loss']+=total_loss.item(); sums['ce']+=(ce_c+ce_p); sums['dice']+=(dice_c+dice_p)
        sums['aux']+=(aux_c+aux_p); sums['cons']+=cons.item(); sums['disc']+=L_dis.item(); sums['batches']+=1

    # Average train metrics
    avg_train = {k:(v/sums['batches']) for k,v in sums.items() if k!='batches'}

    # Validation
    lora_sam.eval(); seg_head.eval(); aux_clfs.eval()
    global_metrics = initialize_metrics(CONFIG['num_classes'], CONFIG['ignore_index'])
    val_loss=0.0; val_batches=0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [Val]"):
            imgs, masks = imgs.to(device), masks.to(device).long()
            out = lora_sam.sam_model.image_encoder.forward_all(imgs, None)
            logits = seg_head(out['backbone_fpn'])
            loss, _, _, _ = compute_loss(logits, masks, loss_fn, CONFIG['num_classes'], CONFIG['ignore_index'])
            val_loss+=loss.item(); val_batches+=1
            preds = torch.argmax(logits,1).cpu().numpy(); m_np=masks.cpu().numpy()
            for p,m in zip(preds,m_np): accumulate_metrics(p,m,global_metrics,CONFIG['num_classes'],CONFIG['ignore_index'])
    val_loss/=val_batches
    tp,fp,fn,inter,uni = global_metrics['tp'],global_metrics['fp'],global_metrics['fn'],global_metrics['intersection'],global_metrics['union']
    iou=[inter[c]/(uni[c]+1e-6) for c in range(CONFIG['num_classes']) if uni[c]>0]
    prec=[tp[c]/(tp[c]+fp[c]+1e-6) for c in range(CONFIG['num_classes']) if tp[c]+fp[c]>0]
    rec=[tp[c]/(tp[c]+fn[c]+1e-6) for c in range(CONFIG['num_classes']) if tp[c]+fn[c]>0]
    f1=[2*p*r/(p+r+1e-6) for p,r in zip(prec,rec) if p+r>0]
    miou=np.mean(iou) if iou else 0; mprec=np.mean(prec) if prec else 0; mrec=np.mean(rec) if rec else 0; mf1=np.mean(f1) if f1 else 0
    comp = miou*metric_w['iou']+mf1*metric_w['f1']+mprec*metric_w['precision']+mrec*metric_w['recall']

    # ------------------------------------------------------------
    # Optimized logging (original format)
    # ------------------------------------------------------------
    lr_bb = backbone_opt.param_groups[0]['lr']; lr_h = head_opt.param_groups[0]['lr']
    lr_s = style_opt.param_groups[0]['lr'] if style_opt else None
    if CONFIG['use_ldp']:
        log_msg = (
            f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], "
            f"Avg Train Loss: {avg_train['loss']:.4f}, "
            f"Avg Val Loss: {val_loss:.4f}, "
            f"Avg CE Loss: {avg_train['ce']:.4f}, "
            f"Avg Dice Loss: {avg_train['dice']:.4f}, "
            f"Avg Aux Loss: {avg_train['aux']:.4f}, "
            f"Avg Consistency Loss: {avg_train['cons']:.4f}, "
            f"Avg Discrepancy Loss: {avg_train['disc']:.4f}, "
            f"Avg IoU: {miou:.4f}, "
            f"Avg F1 Score: {mf1:.4f}, "
            f"Avg Precision: {mprec:.4f}, "
            f"Avg Recall: {mrec:.4f}, "
            f"Composite Score: {comp:.4f}, "
            f"LR Backbone: {lr_bb:.6f}, "
            f"LR Head: {lr_h:.6f}, "
            f"LR Style: {lr_s:.6f}"
        )
    else:
        log_msg = (
            f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], "
            f"Avg Train Loss: {avg_train['loss']:.4f}, "
            f"Avg Val Loss: {val_loss:.4f}, "
            f"Avg CE Loss: {avg_train['ce']:.4f}, "
            f"Avg Dice Loss: {avg_train['dice']:.4f}, "
            f"Avg Aux Loss: {avg_train['aux']:.4f}, "
            f"Avg IoU: {miou:.4f}, "
            f"Avg F1 Score: {mf1:.4f}, "
            f"Avg Precision: {mprec:.4f}, "
            f"Avg Recall: {mrec:.4f}, "
            f"Composite Score: {comp:.4f}, "
            f"LR Backbone: {lr_bb:.6f}, "
            f"LR Head: {lr_h:.6f}"
        )
    logging.info(log_msg)
    print(log_msg)

    if comp > best_score:
        best_score = comp
        suffix = '_lora' + ('_LDP' if CONFIG['use_ldp'] else '')
        lora_sam.save_lora_parameters(os.path.join(w_dir, f"{CONFIG['save_prefix']}{suffix}.safetensors"))
        torch.save(seg_head.state_dict(), os.path.join(w_dir, f"{CONFIG['save_prefix']}{suffix}.pth"))
        print(f"Best model saved at epoch {epoch+1} with score {best_score:.4f}")
        logging.info("Best model checkpointed.")

    # Step schedulers
    bb_sched.step(); h_sched.step();
    if s_sched: s_sched.step()

print("Training finished.")
logging.info("Training finished.")
