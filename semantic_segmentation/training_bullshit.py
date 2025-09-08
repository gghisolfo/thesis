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

# Config
USE_DEEPLAB = False
IMAGE_SIZE = (120, 70)
NUM_CLASSES = 10
BATCH_SIZE = 4
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mappatura esplicita dei valori della maschera (esempio)
LABEL_VALUES = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]  # 10 classi

def map_mask(mask_array):
    mapped = np.zeros_like(mask_array)
    for i, val in enumerate(LABEL_VALUES):
        mapped[mask_array == val] = i
    return mapped



# Ottieni tutte le coppie immagine-maschera
all_images = sorted([os.path.join("./arkanoid_atom/semantic_segmentation/output/images", f) for f in os.listdir("./arkanoid_atom/semantic_segmentation/output/images")])
all_masks = sorted([os.path.join("./arkanoid_atom/semantic_segmentation/output/masks_color", f) for f in os.listdir("./arkanoid_atom/semantic_segmentation/output/masks_color")])

# Dividi in train e val
train_imgs, val_imgs, train_masks, val_masks = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)



# Dataset personalizzato
# class SegmentationDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, transform=None):
#         self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
#         self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert("RGB").resize(IMAGE_SIZE)
#         mask = Image.open(self.mask_paths[idx]).convert("L").resize(IMAGE_SIZE)

#         image = transforms.ToTensor()(image)
#         # mask = torch.from_numpy(np.array(mask)).long()
#         mask_np = np.array(mask)
#         mask_np = map_mask(mask_np)  # <-- Mappatura qui
#         mask = torch.from_numpy(mask_np).long()

#         return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize(IMAGE_SIZE)

        mask = Image.open(self.mask_paths[idx]).convert("L")#.resize(IMAGE_SIZE)
        mask = mask.resize(IMAGE_SIZE, resample=Image.NEAREST)

        image = transforms.ToTensor()(image)
        mask_np = np.array(mask)
        mask_np = map_mask(mask_np)  # Mappa valori a 0-9
        mask = torch.from_numpy(mask_np).long()


        if idx < 3:  # solo primi 3 campioni
            mask_np = np.array(mask)  # maschera come numpy array
            print("Mask before mapping:", mask_np)  # Debug: visualizza prima della mappatura

            mask_np = map_mask(mask_np)  # Mappatura dei valori
            print("Mask after mapping:", mask_np)  # Debug: visualizza dopo la mappatura

            # Salva l'immagine
            Image.fromarray(mask_np.astype(np.uint8)).save(f"check_mask_raw_{idx}.png")
            Image.fromarray((mask_np * 25).astype(np.uint8)).save(f"check_mask_mapped_{idx}.png")


        return image, mask




# Dataset + DataLoader
# dataset = SegmentationDataset(
#     images_dir="./arkanoid_atom/semantic_segmentation/output/images",
#     masks_dir="./arkanoid_atom/semantic_segmentation/output/masks_color"
# )
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

train_dataset = SegmentationDataset(train_imgs, train_masks)
val_dataset = SegmentationDataset(val_imgs, val_masks)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# def show_samples(data_loader, loader_name="Loader", num_samples=4):
#     for images, masks in data_loader:
#         plt.figure(figsize=(12, 3 * num_samples))
#         for i in range(min(num_samples, images.size(0))):
#             image = images[i].permute(1, 2, 0).numpy()  # Converti per matplotlib
#             mask = masks[i].numpy()  # Maschera come array numpy

#             plt.subplot(num_samples, 2, 2*i + 1)
#             plt.imshow(image)
#             plt.title(f"{loader_name} - Image {i}")
#             plt.axis("off")

#             plt.subplot(num_samples, 2, 2*i + 2)
#             plt.imshow(mask, cmap='gray')
#             plt.title(f"{loader_name} - Mask {i}")
#             plt.axis("off")
        
#         plt.tight_layout()
#         plt.show()
#         break  # mostra solo il primo batch




def show_samples(data_loader, loader_name="Loader", num_samples=4):
    # Definisci una colormap con tanti colori quanti sono i label
    class_colors = [
        (0.0, 0.0, 0.0),        # 0: background/environment (nero)
        (1.0, 0.0, 0.0),        # 1: ball (rosso)
        (0.0, 0.0, 1.0),        # 2: paddle_left (blu)
        (0.0, 1.0, 1.0),        # 3: paddle_center (ciano)
        (0.0, 1.0, 0.0),        # 4: paddle_right (verde)
        (1.0, 1.0, 0.0),        # 5: wall_left (giallo)
        (1.0, 0.0, 1.0),        # 6: wall_right (magenta)
        (0.5, 0.5, 0.5),        # 7: wall_top (grigio)
        (1.0, 0.5, 0.0),        # 8: wall_bottom (arancione)
        (0.5, 0.0, 0.5),        # 9+: bricks (viola)
    ]
    cmap = ListedColormap(class_colors)

    for images, masks in data_loader:
        plt.figure(figsize=(12, 3 * num_samples))
        for i in range(min(num_samples, images.size(0))):
            image = images[i].permute(1, 2, 0).numpy()
            mask = masks[i].numpy()

            plt.subplot(num_samples, 2, 2*i + 1)
            plt.imshow(image)
            plt.title(f"{loader_name} - Image {i}")
            plt.axis("off")

            plt.subplot(num_samples, 2, 2*i + 2)
            plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
            plt.title(f"{loader_name} - Mask {i}")
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        break  # Solo il primo batch


# Visualizza campioni dal training e validation set
show_samples(train_loader, loader_name="Train")
show_samples(val_loader, loader_name="Val")



# Modello
model = get_deeplabv3plus_model(3, NUM_CLASSES) if USE_DEEPLAB else UNet(3, NUM_CLASSES)
model = model.to(DEVICE)

# Loss e ottimizzatore
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
# for epoch in range(EPOCHS):
#     print("running epoch: ", epoch)
#     model.train()
#     running_loss = 0.0
#     for images, masks in dataloader:
#         print("     -> elaborating image")
#         images, masks = images.to(DEVICE), masks.to(DEVICE)

#         outputs = model(images)
#         loss = criterion(outputs, masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f}")

# # Salva modello
# torch.save(model.state_dict(), "model.pth")

# Training
for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        
        print("         -> evaluation image")

        images, masks = images.to(DEVICE), masks.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

    acc = correct_pixels / total_pixels
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")




save_dir = "predictions"
os.makedirs(save_dir, exist_ok=True)

CLASS_COLORS = np.array([
    [0, 0, 0],        # Classe 0: nero
    [255, 0, 0],      # Classe 1: rosso
    [0, 255, 0],      # Classe 2: verde
    [0, 0, 255],      # Classe 3: blu
    [255, 255, 0],    # Classe 4: giallo
    [255, 0, 255],    # Classe 5: magenta
    [0, 255, 255],    # Classe 6: ciano
    [128, 128, 128],  # Classe 7: grigio
    [255, 165, 0],    # Classe 8: arancione
    [255, 255, 255],  # Classe 9: bianco
], dtype=np.uint8)


# for i, (images, masks) in enumerate(val_loader):
#     images, masks = images.to(DEVICE), masks.to(DEVICE)

#     outputs = model(images)
#     preds = torch.argmax(outputs, dim=1)

#     for j in range(images.size(0)):
#         pred_mask = preds[j].cpu().numpy().astype(np.uint8)
#         color_pred = CLASS_COLORS[pred_mask]  # (H, W, 3)
#         pred_img = Image.fromarray(color_pred)
#         pred_img.save(os.path.join(save_dir, f"epoch{epoch+1}_img{i*BATCH_SIZE + j}_pred.png"))


for i, (images, masks) in enumerate(val_loader):
    images, masks = images.to(DEVICE), masks.to(DEVICE)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

    for j in range(images.size(0)):
        image = images[j].cpu().permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        true_mask = masks[j].cpu().numpy()
        pred_mask = preds[j].cpu().numpy()

        color_true = CLASS_COLORS[true_mask]
        color_pred = CLASS_COLORS[pred_mask]

        # Visualizza
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(color_true)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(color_pred)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    # Per mostrare solo il primo batch, rompi qui
    break
