
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple

def read_split_files(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        file_names = f.read().strip().split('\n')
    return file_names

class SegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, file_list: List[str],
                 mask_size: Tuple[int, int]=(1024, 1024), is_train: bool=True):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_size = mask_size
        self.file_list = file_list
        self.is_train = is_train

        self.image_files = [
            f for f in os.listdir(self.image_dir)
            if f.endswith('.png') and f.replace('.png', '') in file_list
        ]
        self.mask_files = [
            f for f in os.listdir(self.mask_dir)
            if f.endswith('.png') and f.replace('.png', '') in file_list
        ]
        assert len(self.image_files) == len(self.mask_files), "Image and mask count mismatch."

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
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = self.rgb_loader(image_path)

        mask_file = image_file.replace('.png', '.png')
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = self.binary_loader(mask_path)

        image = self.transform(image)
        mask = self.mask_resize(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))


        if self.is_train:
            if random.random() < 0.5:
                aug_type = random.choice(['horizontal', 'vertical', 'rotation'])
                if aug_type == 'horizontal':
                    image = transforms.functional.hflip(image)
                    mask = transforms.functional.hflip(mask)
                elif aug_type == 'vertical':
                    image = transforms.functional.vflip(image)
                    mask = transforms.functional.vflip(mask)
                elif aug_type == 'rotation':
                    angle = random.choice([90, 180, 270])
                    image = torch.rot90(image, k=angle//90, dims=[1, 2])
                    mask = torch.rot90(mask, k=angle//90, dims=[0, 1])

        return image, mask

    def rgb_loader(self, path):
        return Image.open(path).convert('RGB')

    def binary_loader(self, path):
        return Image.open(path).convert('L')
