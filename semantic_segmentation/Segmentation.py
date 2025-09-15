from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from u_net import UNet
from deep_labv3_plus import get_deeplabv3plus_model
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


# === Colormap per visualizzazione ===
COLOR_MAP = np.array([
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


CLASS_COLORS = COLOR_MAP

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

        # Debug e salvataggio maschere mappate per i primi 3 esempi
        if idx < 3:
            mask_np = np.array(mask)
            print("Mask before mapping:", np.unique(mask_np))

            mask_np = map_mask(mask_np)
            print("Mask after mapping:", np.unique(mask_np))

            print("Classi uniche prima della mappatura:", np.unique(np.array(mask)))
            print("Valori unici nella maschera originale:", np.unique(mask))

            mask_np = map_mask(np.array(mask))
            print("Classi uniche dopo la mappatura:", np.unique(mask_np))


            # Image.fromarray(mask_np.astype(np.uint8)).save(f"check_mask_raw_{idx}.png")
            # Image.fromarray((mask_np * 25).astype(np.uint8)).save(f"check_mask_mapped_{idx}.png")

            # mask = Image.fromarray(mask_np.astype(np.uint8))

        # mask = Image.fromarray(mask_np.astype(np.uint8))

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