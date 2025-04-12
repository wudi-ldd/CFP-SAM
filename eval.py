# Import necessary libraries
import os
import torch
import torch.nn as nn
import numpy as np
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
from sam2.build_sam import build_sam2  # Import SAM2 model builder
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from safetensors.torch import load_file  # For loading safetensors files
from lora import LoRA_sam2  # Ensure LoRA_sam2 class is correctly defined in lora.py
from types import MethodType

# Configuration parameters
CONFIG = {
    'dataset_name': 'Vaihingen',          # Dataset name
    'test_files': 'val.txt',             # File list
    'base_dir': 'logs',                   # Base directory
    'log_file': 'test_results.log',       # Log file name
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Device selection
    'num_classes': 5,                      # Number of classes, 0: background, 1-6: foreground classes
    'ignore_index': 255,                   # Ignored class index
    'save_prefix': 'best_model',           # Prefix for weight file names

    # Overlay configuration
    'save_overlay': True,                  # Whether to save overlay images
    'overlay_mode': 'save_separate_masks', # Overlay mode: 'save_overlaid_images' or 'save_separate_masks'
    'overlay_alpha': 0.2,                  # Overlay transparency
    'overlay_dir': 'predictions',          # Directory to save overlay results

    'image_size': (1024, 1024),            # Size of input and mask images
    'lora_rank': 256,                     # Rank of LoRA
    'lora_alpha': 256,                    # Scaling factor of LoRA
    'checkpoint': "weights/sam2.1_hiera_large.pt",  # Model checkpoint path
    'model_cfg': "configs/sam2.1/sam2.1_hiera_l.yaml",  # Model configuration file path

    'use_ldp': True                        # Whether to use LDP mode for evaluation
}

# Create base directory structure
base_log_dir = os.path.join(CONFIG['base_dir'], CONFIG['dataset_name'])
logs_dir = os.path.join(base_log_dir, 'logs')
weights_dir = os.path.join(base_log_dir, 'weights')
predictions_dir = os.path.join(base_log_dir, CONFIG['overlay_dir'])

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(logs_dir, CONFIG['log_file']),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Force reconfigure logger (applicable for Python 3.8+)
)

# Set device
device = CONFIG['device']

# Define evaluation metric functions
def initialize_metrics(num_classes):
    # Initialize evaluation metrics
    return {
        'tp': [0] * num_classes,           # True positives
        'fp': [0] * num_classes,           # False positives
        'fn': [0] * num_classes,           # False negatives
        'intersection': [0] * num_classes, # Intersection
        'union': [0] * num_classes,        # Union
        'correct': 0,                       # Number of correctly predicted pixels
        'total': 0                          # Total number of pixels
    }

def accumulate_metrics(preds, targets, global_metrics, num_classes, ignore_index=255):
    # Ignore pixels with label ignore_index
    valid = (targets != ignore_index)
    preds = preds[valid]
    targets = targets[valid]

    # Accumulate the number of correctly predicted pixels and total pixels
    global_metrics['correct'] += (preds == targets).sum()
    global_metrics['total'] += valid.sum()

    # Accumulate evaluation metrics for each class
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

# Define SegmentationHead (without weighted overlay)
class SegmentationHead(nn.Module):
    """
    Segmentation head, used to generate the final segmentation result by fusing all FPN feature maps.
    """
    def __init__(self, fpn_channels, out_channels, align_corners=False):
        super(SegmentationHead, self).__init__()
        self.align_corners = align_corners

        # Define 3x3 convolutions for each FPN feature map for smoothing
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(ch, 256, kernel_size=3, padding=1) for ch in fpn_channels
        ])

        # Define the final convolutional layer after fusion
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

    def forward(self, fpn_features):
        """
        Forward pass.

        Parameters:
            fpn_features (list of Tensor): List of FPN feature maps, shaped as [(B, C, H, W), ...].

        Returns:
            Tensor: Output logits, shaped as (B, num_classes, CONFIG['image_size'][0], CONFIG['image_size'][1]).
        """
        # Assume the first feature map has the highest resolution
        target_size = fpn_features[0].shape[2:]  # (H, W)
        fused_features = torch.zeros(
            (fpn_features[0].shape[0], 256, target_size[0], target_size[1]),
            device=fpn_features[0].device
        )

        for conv, feature in zip(self.smooth_convs, fpn_features):
            x = conv(feature)
            # Upsample the feature map to the target size
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
            # Directly add without weighting
            fused_features += x  # Direct addition

        # Pass through the final convolutional layer
        x = self.final_conv(fused_features)

        # Upsample the output to a fixed size, e.g., CONFIG['image_size']
        x = F.interpolate(x, size=CONFIG['image_size'], mode='bilinear', align_corners=self.align_corners)
        return x

# Define forward_inter method for the image encoder
def forward_inter(self, sample: torch.Tensor):
    # Forward pass through the trunk network
    trunk_features = self.trunk(sample)
    features, pos = self.neck(trunk_features)

    src = features[-1]
    output = {
        "vision_features": src,
        "vision_pos_enc": pos,
        "backbone_fpn": features,
        "trunk_features": trunk_features,  # Corrected to trunk_features
    }
    return output

# Build and configure SAM2 model
checkpoint = CONFIG['checkpoint']
model_cfg = CONFIG['model_cfg']
sam_model = build_sam2(model_cfg, checkpoint, device)
sam_model.image_encoder.forward_inter = MethodType(forward_inter, sam_model.image_encoder)

# Remove unnecessary components
components_to_remove = [
    'sam_mask_decoder',
    'sam_prompt_encoder',
    'memory_encoder',
    'memory_attention',
    'mask_downsample',
    'obj_ptr_tpos_proj',
    'obj_ptr_proj'
]
for component in components_to_remove:
    if hasattr(sam_model, component):
        delattr(sam_model, component)

# Initialize LoRA_sam2 model using parameters from CONFIG
lora_sam_model = LoRA_sam2(
    sam_model, 
    rank=CONFIG['lora_rank'], 
    alpha=CONFIG['lora_alpha']
)
lora_sam_model.to(device)
lora_sam_model.eval()  # Set LoRA model to evaluation mode

# Build the suffix for the weight file based on use_ldp option
lora_suffix = "_lora_LDP" if CONFIG['use_ldp'] else "_lora"

# Load LoRA parameters
lora_path = os.path.join(weights_dir, f"{CONFIG['save_prefix']}{lora_suffix}.safetensors")
if os.path.exists(lora_path):
    lora_sam_model.load_lora_parameters(lora_path)
    print(f"Loaded LoRA parameters from {lora_path}")
    logging.info(f"Loaded LoRA parameters from {lora_path}")
else:
    raise FileNotFoundError(f"Could not find LoRA parameter file at {lora_path}")

# Create SegmentationHead instance and load weights
seg_head_path = os.path.join(weights_dir, f"{CONFIG['save_prefix']}{lora_suffix}.pth")
model_seg_head = SegmentationHead(
    fpn_channels=[256, 256, 256, 256],    # Number of channels in backbone_fpn feature maps
    out_channels=CONFIG['num_classes'],    # Number of output classes
    align_corners=False                   # Align corners parameter
)
model_seg_head.to(device)
if os.path.exists(seg_head_path):
    model_seg_head.load_state_dict(torch.load(seg_head_path, map_location=device))
    model_seg_head.eval()  # Set SegmentationHead to evaluation mode
    print(f"Loaded SegmentationHead weights from {seg_head_path}")
    logging.info(f"Loaded SegmentationHead weights from {seg_head_path}")
else:
    raise FileNotFoundError(f"Could not find SegmentationHead weight file at {seg_head_path}")

# Define dataset class
def read_split_files(file_path):
    with open(file_path, 'r') as f:
        file_names = f.read().strip().split('\n')
    return file_names

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, mask_size=(1024, 1024), is_train=True):
        """
        Args:
            image_dir (str): Path to the image directory.
            mask_dir (str): Path to the mask directory.
            file_list (list): List of file names.
            mask_size (tuple): Size of the masks.
            is_train (bool): Whether it is in training mode, used for data augmentation.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_size = mask_size
        self.file_list = file_list
        self.is_train = is_train

        # Filter image files that meet the conditions
        self.image_files = [
            f for f in os.listdir(self.image_dir)
            if f.endswith('.png') and f.replace('.png', '') in file_list
        ]
        self.mask_files = [
            f for f in os.listdir(self.mask_dir)
            if f.endswith('.png') and f.replace('.png', '') in file_list
        ]

        # Ensure the number of images and masks match
        assert len(self.image_files) == len(self.mask_files), "Image and mask count mismatch."

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((mask_size[0], mask_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_resize = transforms.Resize((mask_size[0], mask_size[1]), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Read image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = self.rgb_loader(image_path)

        # Read mask, file name remains the same, only extension differs
        mask_file = image_file.replace('.png', '.png')
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = self.binary_loader(mask_path)

        # Apply transforms
        image = self.transform(image)
        mask = self.mask_resize(mask)

        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # Convert to Tensor

        # Data augmentation: perform horizontal and vertical flips in training mode
        if self.is_train:
            if random.random() > 0.5:  # 50% probability of horizontal flip
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:  # 50% probability of vertical flip
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

        return image, mask, image_file  # Return image_file for later saving

    def rgb_loader(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def binary_loader(self, path):
        img = Image.open(path).convert('L')
        return img

# Prepare test dataset and data loader
dataset_name = CONFIG['dataset_name']
image_dir = os.path.join('datasets', dataset_name, 'images')
mask_dir = os.path.join('datasets', dataset_name, 'masks')
test_files = read_split_files(os.path.join('datasets', dataset_name, CONFIG['test_files']))

test_dataset = SegmentationDataset(
    image_dir, mask_dir, test_files, mask_size=CONFIG['image_size'], is_train=False
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

# Define color map
PRED_COLOR_MAP = {
    0: (240, 240, 240),  # Background: light gray
    1: (179, 215, 255),  # Sky blue
    2: (156, 226, 206),  # Mint green
    3: (255, 214, 184),  # Apricot
    4: (212, 188, 255),  # Lavender
    5: (189, 245, 198),  # Light green
    6: (255, 192, 218),  # Light pink
    7: (255, 235, 174),  # Pale yellow
    8: (218, 199, 180),  # Linen
    9: (177, 230, 245),  # Light cyan
    10: (232, 184, 255), # Light purple
    11: (210, 230, 175), # Bud green
    12: (255, 175, 165), # Light coral
}

# Initialize global evaluation metrics
global_metrics = initialize_metrics(CONFIG['num_classes'])

# Calculate total number of samples
total_samples = len(test_dataset)

# Evaluation loop
with torch.no_grad():
    for batch_idx, (images, masks, image_files) in enumerate(tqdm(test_loader, desc="Testing")):
        images, masks = images.to(device), masks.to(device).long()

        # Get the output of the image encoder
        image_embedding = lora_sam_model.sam_model.image_encoder.forward_inter(images)
        backbone_fpn_features = image_embedding['backbone_fpn']  # List of 4 FPN feature maps

        # Get the output of all FPN feature maps through the segmentation head
        class_logits = model_seg_head(backbone_fpn_features)

        # Get predicted classes
        preds = torch.argmax(class_logits, dim=1).long()

        # Convert predictions and masks to numpy arrays
        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()

        # Accumulate global evaluation metrics
        for pred, mask in zip(preds_np, masks_np):
            accumulate_metrics(pred, mask, global_metrics, CONFIG['num_classes'], ignore_index=CONFIG['ignore_index'])

        # If saving overlay images is enabled
        if CONFIG['save_overlay']:
            # Get image file name
            image_file = image_files[0]
            image_path = os.path.join(image_dir, image_file)

            if CONFIG['overlay_mode'] == 'save_overlaid_images':
                # Read original image
                original_image = cv2.imread(image_path)
                if original_image is None:
                    logging.error(f"Unable to read image: {image_path}")
                    continue
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                original_image = cv2.resize(original_image, CONFIG['image_size'], interpolation=cv2.INTER_NEAREST)

                # Generate overlay image for predicted mask
                mask_color_pred = np.zeros_like(original_image, dtype=np.uint8)
                mask_color_gt = np.zeros_like(original_image, dtype=np.uint8)
                for cls, color in PRED_COLOR_MAP.items():
                    mask_color_pred[preds_np[0] == cls] = color
                    mask_color_gt[masks_np[0] == cls] = color

                # Create overlay image: predicted mask
                overlay_pred = cv2.addWeighted(original_image, 1 - CONFIG['overlay_alpha'],
                                              mask_color_pred, CONFIG['overlay_alpha'], 0)

                # Create overlay image: ground truth mask
                overlay_gt = cv2.addWeighted(original_image, 1 - CONFIG['overlay_alpha'],
                                            mask_color_gt, CONFIG['overlay_alpha'], 0)

                # Save overlay images
                save_file_pred = os.path.splitext(image_file)[0] + '_pred_overlay.png'
                save_path_pred = os.path.join(predictions_dir, save_file_pred)
                overlay_bgr_pred = cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path_pred, overlay_bgr_pred)

                save_file_gt = os.path.splitext(image_file)[0] + '_gt_overlay.png'
                save_path_gt = os.path.join(predictions_dir, save_file_gt)
                overlay_bgr_gt = cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path_gt, overlay_bgr_gt)

            elif CONFIG['overlay_mode'] == 'save_separate_masks':
                # Generate predicted mask image
                mask_color_pred = np.zeros((*CONFIG['image_size'], 3), dtype=np.uint8)
                mask_color_gt = np.zeros((*CONFIG['image_size'], 3), dtype=np.uint8)
                for cls, color in PRED_COLOR_MAP.items():
                    mask_color_pred[preds_np[0] == cls] = color
                    mask_color_gt[masks_np[0] == cls] = color

                # Save predicted mask
                save_file_pred = os.path.splitext(image_file)[0] + '_pred_mask.png'
                mask_bgr_pred = cv2.cvtColor(mask_color_pred, cv2.COLOR_RGB2BGR)
                save_path_pred = os.path.join(predictions_dir, save_file_pred)
                cv2.imwrite(save_path_pred, mask_bgr_pred)

                # Save ground truth mask
                save_file_gt = os.path.splitext(image_file)[0] + '_gt_mask.png'
                mask_bgr_gt = cv2.cvtColor(mask_color_gt, cv2.COLOR_RGB2BGR)
                save_path_gt = os.path.join(predictions_dir, save_file_gt)
                cv2.imwrite(save_path_gt, mask_bgr_gt)

            else:
                raise ValueError("Invalid overlay_mode. Choose 'save_overlaid_images' or 'save_separate_masks'")

# Calculate evaluation metrics
all_classes = list(range(CONFIG['num_classes']))

for cls in all_classes:
    intersection = global_metrics['intersection'][cls]
    union = global_metrics['union'][cls]
    tp = global_metrics['tp'][cls]
    fp = global_metrics['fp'][cls]
    fn = global_metrics['fn'][cls]

    iou = intersection / (union + 1e-6) if union > 0 else 0
    precision = tp / (tp + fp + 1e-6) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn + 1e-6) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0

    print(f"Class {cls} - IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logging.info(f"Class {cls} - IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Calculate average metrics for all classes
valid_iou = [global_metrics['intersection'][cls] / (global_metrics['union'][cls] + 1e-6) 
             for cls in all_classes if global_metrics['union'][cls] > 0]
avg_iou = np.mean(valid_iou) if valid_iou else 0

valid_precision = [global_metrics['tp'][cls] / (global_metrics['tp'][cls] + global_metrics['fp'][cls] + 1e-6) 
                   for cls in all_classes if (global_metrics['tp'][cls] + global_metrics['fp'][cls]) > 0]
avg_precision = np.mean(valid_precision) if valid_precision else 0

valid_recall = [global_metrics['tp'][cls] / (global_metrics['tp'][cls] + global_metrics['fn'][cls] + 1e-6) 
                for cls in all_classes if (global_metrics['tp'][cls] + global_metrics['fn'][cls]) > 0]
avg_recall = np.mean(valid_recall) if valid_recall else 0

valid_f1 = [2 * (tp / (tp + fp + 1e-6)) * (tp / (tp + fn + 1e-6)) / 
            ((tp / (tp + fp + 1e-6)) + (tp / (tp + fn + 1e-6)) + 1e-6)
            for cls, tp, fp, fn in zip(all_classes, global_metrics['tp'], global_metrics['fp'], global_metrics['fn']) 
            if (tp + fp) > 0 and (tp + fn) > 0]
avg_f1 = np.mean(valid_f1) if valid_f1 else 0

# Calculate Overall Accuracy (OA)
oa = global_metrics['correct'] / global_metrics['total'] if global_metrics['total'] > 0 else 0

print(f"Average metrics - IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, OA: {oa:.4f}")
logging.info(f"Average metrics - IoU: {avg_iou:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, OA: {oa:.4f}")