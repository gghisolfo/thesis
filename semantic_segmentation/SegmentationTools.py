
from PIL import Image
import numpy as np
import os

import torch
from torch.utils.data import Dataset 
from torchvision import transforms


# === Colormap per visualizzazione ===
CLASS_COLORS_ORIGINAL = np.array([
    [0, 0, 0],         # 0 - background - environment (sfondo) - nero
    [255, 0, 0],       # 1 - ball - ROSSO
    [0, 0, 255],       # 2 - paddle_left - blu pieno
    [0, 100, 255],     # 3 - paddle_center - blu medio-chiaro
    [0, 150, 255],     # 4 - paddle_right - blu tendente al ciano
    [0, 255, 0],       # 5 - wall_left - verde acceso
    [0, 255, 50],      # 6 - wall_right - verde acesso
    [0, 255, 150],     # 7 - wall_top - acquamarina
    [0, 255, 150],     # 8 - wall_bottom - acquamarina
    [255, 255, 255]    # 9 - bricks - bianco
], dtype=np.uint8)


CLASS_COLORS_DIFFERENT = np.array([
    [10, 10, 10],       # 0 - environment (sfondo) → grigio molto scuro, non puro nero
    [255, 60, 60],      # 1 - ball → rosso brillante (attira l’attenzione)
    [60, 100, 255],     # 2 - paddle_left → blu intenso
    [60, 180, 255],     # 3 - paddle_center → azzurro vivace
    [80, 255, 255],     # 4 - paddle_right → ciano chiaro
    [60, 255, 100],     # 5 - wall_left → verde acceso
    [140, 255, 60],     # 6 - wall_right → verde lime / giallo-verde
    [255, 200, 60],     # 7 - wall_top → giallo-arancio caldo
    [255, 120, 60],     # 8 - wall_bottom → arancione più intenso
    [255, 255, 255],    # 9 - bricks → bianco pieno, massimo contrasto
], dtype=np.uint8)


def map_mask(mask):
    mapped_mask = np.zeros_like(mask)
    mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
    }
    for original_value, class_index in mapping.items():
        mapped_mask[mask == original_value] = class_index

    return mapped_mask

# === Data Augmentation semplice ===
def augment(img, mask):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform  # aggiunto

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carica immagine e maschera
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Maschera in scala di grigi

        mask_np = map_mask(np.array(mask))
     

        # Applica trasformazioni
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.PILToTensor()(mask).squeeze(0).long()  # shape [H, W], dtype: long

        return image, mask