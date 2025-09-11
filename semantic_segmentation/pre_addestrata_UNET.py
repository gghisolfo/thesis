import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import os

import matplotlib.pyplot as plt

# =========================
# CONFIGURAZIONE
# =========================
num_classes = 10           # cambia in base al tuo dataset
image_size = (256, 256)   # input U-Net
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
test_split = 0.2          # percentuale immagini da tenere per test


# =========================
# DATASET PERSONALIZZATO
# =========================
class BioDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        # Carica immagine e maschera
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        # Ridimensiona
        img = img.resize(image_size)
        mask = mask.resize(image_size, resample=Image.NEAREST)

        # Applica eventuale augment
        if self.transform:
            img, mask = self.transform(img, mask)

        # Preprocessing immagine
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)

        # Converte maschera RGB → classi
        mask = np.array(mask)                      # [H, W, 3]
        mask = mask_rgb_to_class(mask)             # [H, W] con indici 0..num_classes-1
        mask = torch.from_numpy(mask).long()       # Tensor [H, W]

        return img, mask


# =========================
# DATA AUGMENTATION SEMPLICE
# =========================
def augment(img, mask):
    # flip casuale
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

# =========================
# MAPPATURA COLORI -> CLASSI
# =========================
color2id = {
    (0, 0, 0): 0,          # nero
    (255, 0, 0): 1,        # rosso
    (0, 255, 0): 2,        # verde
    (0, 0, 255): 3,        # blu
    (255, 255, 0): 4,      # giallo
    (0, 255, 255): 5,      # ciano
    (255, 0, 255): 6,      # magenta
    (255, 165, 0): 7,      # arancione
    (128, 0, 128): 8,      # viola
    (192, 192, 192): 9     # grigio chiaro
}

# Converte una maschera RGB in maschera di indici [H,W].
def mask_rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    mask_class = np.zeros((h, w), dtype=np.int64)

    for color, class_id in color2id.items():
        matches = np.all(mask_rgb == color, axis=-1)
        mask_class[matches] = class_id

    return mask_class


# =========================
# PREPARA DATASET E DATALOADER
# =========================
# Sostituisci con i tuoi percorsi

# image_path1 = 'C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/output/images/frame_0008.png'
# image_path2 = 'C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/output/images/frame_0009.png'
# mask_path1 = 'C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/output/masks_color/frame_0008.png'
# mask_path2 = 'C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/output/masks_color/frame_0009.png'
# train_images = [image_path1, image_path2]
# train_masks =  [mask_path1, mask_path2]
# dataset = BioDataset(train_images, train_masks, transform=augment)
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


images_dir = "C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/dataset/images"
masks_dir  = "C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/dataset/masks_color"

image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".png")])
mask_files  = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".png")])

assert len(image_files) == len(mask_files), "Numero immagini e maschere non corrisponde!"

# Dividi in train e test
total_size = len(image_files)
test_size = int(total_size * test_split)
train_size = total_size - test_size

train_images, test_images = image_files[:train_size], image_files[train_size:]
train_masks,  test_masks  = mask_files[:train_size],  mask_files[test_size:]

train_dataset = BioDataset(train_images, train_masks, transform=augment)
test_dataset  = BioDataset(test_images, test_masks, transform=None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train: {len(train_dataset)} immagini, Test: {len(test_dataset)} immagini")


# =========================
# CREA MODELLO U-NET
# =========================
model = smp.Unet(
    encoder_name="resnet34",        # encoder pre-addestrato
    encoder_weights="imagenet",
    classes=num_classes,
    activation=None               # logits, usa CrossEntropyLoss
).to(device)

# Congela encoder se vuoi ridurre overfitting
for param in model.encoder.parameters():
    param.requires_grad = False  # False per congelare -- prima: True

# =========================
# LOSS E OTTIMIZZATORE
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #prima 1e-4

# =========================
# TRAINING LOOP SEMPLICE
# =========================
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        # Rimuove canale inutile se presente
        if masks.dim() == 4:   # [B,1,H,W] -> [B,H,W]
            masks = masks.squeeze(1)

        optimizer.zero_grad()
        outputs = model(imgs)  # [B,C,H,W]

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

# =========================
# INFERENZA SU UNA IMMAGINE
# =========================
model.eval()

# test_img_path = "C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/output/images/frame_0010.png"

test_img_path = test_images[0]

img = Image.open(test_img_path).convert("RGB") #test image
img_resized = img.resize(image_size)
input_tensor = transforms.ToTensor()(img_resized)
input_tensor = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(input_tensor)
input_tensor = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    seg_map = torch.argmax(output, dim=1)[0].cpu().numpy()  # [H,W] test image map

# Ridimensiona al formato originale
seg_map_orig = np.array(Image.fromarray(seg_map.astype(np.uint8)).resize(img.size, resample=Image.NEAREST))

# Colori di esempio
# colors = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255],[255,255,0]], dtype=np.uint8)

colors = np.array([
    [0, 0, 0],        # nero
    [255, 0, 0],      # rosso
    [0, 255, 0],      # verde
    [0, 0, 255],      # blu
    [255, 255, 0],    # giallo
    [0, 255, 255],    # ciano
    [255, 0, 255],    # magenta
    [255, 165, 0],    # arancione
    [128, 0, 128],    # viola
    [192, 192, 192]   # grigio chiaro
], dtype=np.uint8)

seg_rgb = colors[seg_map_orig]

# =========================
# VALUTAZIONE mIoU
# =========================
def evaluate_miou(model, dataloader, num_classes):
    model.eval()
    iou_per_class = np.zeros(num_classes)
    count_per_class = np.zeros(num_classes)

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            for c in range(num_classes):
                pred_c = (preds == c)
                mask_c = (masks == c)

                intersection = np.logical_and(pred_c, mask_c).sum()
                union = np.logical_or(pred_c, mask_c).sum()

                if union > 0:
                    iou_per_class[c] += intersection / union
                    count_per_class[c] += 1

    iou_per_class = np.divide(iou_per_class, count_per_class, out=np.zeros_like(iou_per_class), where=count_per_class > 0)
    miou = np.mean(iou_per_class[count_per_class > 0])
    return miou, iou_per_class




def evaluate_miou_and_visualization(model, dataloader, num_classes, device="cpu"):
    print("valutando ... ")
    model.eval()
    intersection_per_class = np.zeros(num_classes, dtype=np.float64)
    union_per_class = np.zeros(num_classes, dtype=np.float64)
    max_samples = 3

    with torch.no_grad():
        # for imgs, masks in dataloader:


        for i, (imgs, masks) in enumerate(dataloader):
            if max_samples is not None and i >= max_samples:
                break  # esce dal loop dopo max_samples batch

            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            # Prendiamo il primo elemento del batch per visualizzazione
            img_vis = imgs[0].detach().cpu().permute(1, 2, 0).numpy()
            mask_vis = masks[0].detach().cpu().numpy()
            pred_vis = torch.argmax(outputs, dim=1)[0].cpu().numpy()

            # Normalizzazione immagine per [0,1] (se necessario)
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)

            print("risultato grafico mostrato:")

            # plt.figure(figsize=(15, 5))
            # plt.subplot(1, 3, 1)
            # plt.imshow(img_vis)
            # plt.axis('off')
            # plt.title("Originale")

            # plt.subplot(1, 3, 2)
            # plt.imshow(mask_vis, cmap="nipy_spectral")
            # plt.axis('off')
            # plt.title("Ground Truth")

            # plt.subplot(1, 3, 3)
            # plt.imshow(pred_vis, cmap="nipy_spectral")
            # plt.axis('off')
            # plt.title("Predizione")
            # plt.show()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_vis)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(mask_vis)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_vis)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

            # Calcolo metriche
            print("CALCOLO metriche ... ")
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            for c in range(num_classes):
                pred_c = (preds == c)
                mask_c = (masks == c)

                intersection = np.logical_and(pred_c, mask_c).sum()
                union = np.logical_or(pred_c, mask_c).sum()

                intersection_per_class[c] += intersection
                union_per_class[c] += union

    # Calcolo IoU per classe
    iou_per_class = intersection_per_class / (union_per_class + 1e-8)
    # mIoU solo sulle classi presenti
    valid_classes = union_per_class > 0
    miou = np.mean(iou_per_class[valid_classes])

    return miou, iou_per_class



print("valutando _ out1")
#miou, iou_classes = evaluate_miou(model, test_loader, num_classes)
miou, iou_classes = evaluate_miou_and_visualization(model, test_loader, num_classes, device)
print("valutando _ out2")


print(f"\nTest mIoU: {miou:.4f}")
for i, val in enumerate(iou_classes):
    print(f"Classe {i}: IoU = {val:.4f}")


# # Visualizzazione
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.axis('off')
# plt.title("Originale")
# plt.subplot(1,2,2)
# plt.imshow(seg_rgb)
# plt.axis('off')
# plt.title("Segmentazione")
# plt.show()


# =========================
# INFERENZA SU UNA IMMAGINE
# =========================

save_model=False
if save_model:
    torch.save(model.state_dict(), "unet_weights.pth")


