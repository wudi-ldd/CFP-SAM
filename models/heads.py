import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG

class SegmentationHead(nn.Module):

    def __init__(self, fpn_channels, out_channels, align_corners=False, gn_groups=32):
        super(SegmentationHead, self).__init__()
        self.align_corners = align_corners

        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(ch, 256, kernel_size=3, padding=1) for ch in fpn_channels
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

    def forward(self, fpn_features):
        target_size = fpn_features[0].shape[2:]

        fused_dtype = fpn_features[0].dtype
        fused_device = fpn_features[0].device

        fused_features = torch.zeros(
            (fpn_features[0].shape[0], 256, target_size[0], target_size[1]),
            device=fused_device,
            dtype=fused_dtype
        )

        for conv, feature in zip(self.smooth_convs, fpn_features):
            x = conv(feature)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=self.align_corners)
            fused_features += x

        x = self.final_conv(fused_features)
        x = F.interpolate(x, size=CONFIG['image_size'], mode='bilinear', align_corners=self.align_corners)
        return x

class AuxiliaryClassifier(nn.Module):

    def __init__(self, in_channels, num_classes, output_size=(1024, 1024), gn_groups=32):
        super(AuxiliaryClassifier, self).__init__()
        self.output_size = output_size

        self.aux_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.aux_gn1 = nn.GroupNorm(gn_groups, 256)
        self.aux_relu1 = nn.ReLU(inplace=True)
        self.aux_conv2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.aux_conv1(x)
        x = self.aux_gn1(x)
        x = self.aux_relu1(x)
        x = self.aux_conv2(x)
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x
