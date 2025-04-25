import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

__all__ = ["read_split_files", "SegmentationDataset"]


def read_split_files(file_path: str):
    """Read the txt split file and return the list of image ids."""
    with open(file_path, "r") as f:
        return f.read().strip().split("\n")


class SegmentationDataset(Dataset):
    """A simple dataset for (image, mask) pairs stored as .png files."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        file_list,
        mask_size=(1024, 1024),
        is_train: bool = True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_size = mask_size
        self.is_train = is_train

        # --------------------------------------------------------------
        # Build file lists
        # --------------------------------------------------------------
        self.image_files = [
            f for f in os.listdir(image_dir) if f.endswith(".png") and f[:-4] in file_list
        ]
        self.mask_files = [
            f for f in os.listdir(mask_dir) if f.endswith(".png") and f[:-4] in file_list
        ]
        assert len(self.image_files) == len(
            self.mask_files
        ), "Image / mask count mismatch!"

        # --------------------------------------------------------------
        # Transforms
        # --------------------------------------------------------------
        self.transform = transforms.Compose(
            [
                transforms.Resize(mask_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_resize = transforms.Resize(mask_size, interpolation=Image.NEAREST)

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    @staticmethod
    def _rgb_loader(path):
        return Image.open(path).convert("RGB")

    @staticmethod
    def _binary_loader(path):
        return Image.open(path).convert("L")

    # --------------------------------------------------------------
    # Core API
    # --------------------------------------------------------------
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = self._rgb_loader(img_path)
        mask = self._binary_loader(mask_path)

        image = self.transform(image)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # ----------------------------------------------------------
        # Light augmentations
        # ----------------------------------------------------------
        if self.is_train and random.random() < 0.5:
            aug_type = random.choice(["hflip", "vflip", "rot"])
            if aug_type == "hflip":
                image = torch.flip(image, [2])
                mask = torch.flip(mask, [1])
            elif aug_type == "vflip":
                image = torch.flip(image, [1])
                mask = torch.flip(mask, [0])
            else:  # rotation
                k = random.choice([1, 2, 3])  # 90, 180, 270 deg
                image = torch.rot90(image, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[0, 1])

        return image, mask
