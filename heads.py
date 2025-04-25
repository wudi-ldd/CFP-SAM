import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG

__all__ = ["SegmentationHead", "AuxiliaryClassifier"]


class SegmentationHead(nn.Module):
    """Simple FPN‑style fusion head without learnable weights between levels."""

    def __init__(self, fpn_channels, out_channels, align_corners=False):
        super().__init__()
        self.align_corners = align_corners

        self.smooth_convs = nn.ModuleList(
            [nn.Conv2d(ch, 256, 3, padding=1) for ch in fpn_channels]
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, out_channels, 1),
        )

    def forward(self, fpn_features):
        target_size = fpn_features[0].shape[2:]
        fused = torch.zeros(
            (fpn_features[0].shape[0], 256, *target_size), device=fpn_features[0].device
        )
        for conv, feat in zip(self.smooth_convs, fpn_features):
            x = conv(feat)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=self.align_corners)
            fused = fused + x
        x = self.final_conv(fused)
        x = F.interpolate(x, size=CONFIG["image_size"], mode="bilinear", align_corners=self.align_corners)
        return x


class AuxiliaryClassifier(nn.Module):
    """Aux branch used only for deep supervision during training."""

    def __init__(self, in_channels, num_classes, output_size=(1024, 1024)):
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        x = self.net(x)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return x
