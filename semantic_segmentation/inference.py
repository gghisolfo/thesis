import matplotlib.pyplot as plt
from u_net import UNet
from segmentation import SegmentationDataset, CLASS_COLORS
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

# ----------------------- Config -----------------------
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1          # batch tipico per inference
USE_DEEPLAB = False     # False se stai usando UNet
MAX_IMAGES = 5          # numero massimo di immagini da visualizzare

# ----------------------- Modello -----------------------
print("Ricrea modello...")
model = UNet(3, NUM_CLASSES)
model = model.to(DEVICE)

print("Carica pesi salvati...")
model.load_state_dict(torch.load("segmentation_model.pth", map_location=DEVICE))
model.eval()  # modalità evaluation

# ----------------------- DataLoader -----------------------
print("Caricamento immagini test...")
test_images_path = "./dataset/test/images"
test_masks_path = "./dataset/test/masks"

test_imgs = sorted([os.path.join(test_images_path, f) for f in os.listdir(test_images_path)])
test_masks = sorted([os.path.join(test_masks_path, f) for f in os.listdir(test_masks_path)])

test_dataset = SegmentationDataset(test_imgs, test_masks)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------- Inferenza -----------------------
all_preds = []
all_masks = []
all_images = []

with torch.no_grad():  # disabilita il calcolo dei gradienti
    for images, masks in test_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(images)['out'] if USE_DEEPLAB else model(images)
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]

        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())
        all_images.append(images.cpu())

# ----------------------- Visualizzazione -----------------------
print("Visualizzazione predizioni...")
count = 0
for batch_idx in range(len(all_preds)):
    batch_preds = all_preds[batch_idx]
    batch_masks = all_masks[batch_idx]
    batch_images = all_images[batch_idx]

    for b in range(batch_preds.size(0)):
        if count >= MAX_IMAGES:
            break

        img = batch_images[b].permute(1,2,0).numpy()  # [H,W,C]
        img = np.clip(img, 0, 1)                      # se normalizzata

        true_mask = batch_masks[b].numpy()
        pred_mask = batch_preds[b].numpy()

        color_true = CLASS_COLORS[true_mask]
        color_pred = CLASS_COLORS[pred_mask]

        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].imshow(img)
        axes[0].set_title('Input')
        axes[0].axis('off')

        axes[1].imshow(color_true)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(color_pred)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        count += 1
    if count >= MAX_IMAGES:
        break
