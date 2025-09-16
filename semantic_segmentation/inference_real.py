import matplotlib.pyplot as plt
from u_net import UNet
from segmentation import CLASS_COLORS
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

# ----------------------- Config -----------------------
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1          # batch tipico per inference
USE_DEEPLAB = False     # False se stai usando UNet
MAX_IMAGES = 5          # numero massimo di immagini da visualizzare
IMAGES_FOLDER = "./dataset/real_images"  # cartella con le immagini da inferire

# ----------------------- Dataset custom per immagini singole -----------------------
class InferenceDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0
        return image, self.image_paths[idx]  # restituisce anche il path per identificare l’immagine

# ----------------------- Modello -----------------------
print("Ricrea modello...")
model = UNet(3, NUM_CLASSES)
model = model.to(DEVICE)

print("Carica pesi salvati...")
model.load_state_dict(torch.load("segmentation_model.pth", map_location=DEVICE))
model.eval()

# ----------------------- DataLoader -----------------------
dataset = InferenceDataset(IMAGES_FOLDER)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------- Inferenza -----------------------
print("Eseguo inferenza...")
count = 0
with torch.no_grad():
    for images, paths in loader:
        images = images.to(DEVICE)
        outputs = model(images)['out'] if USE_DEEPLAB else model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # [B,H,W]

        for b in range(images.size(0)):
            if count >= MAX_IMAGES:
                break

            img = images[b].cpu().permute(1,2,0).numpy()  # [H,W,C]
            img = np.clip(img, 0, 1)

            pred_mask = preds[b]
            color_pred = CLASS_COLORS[pred_mask]

            # Visualizzazione
            fig, axes = plt.subplots(1,2, figsize=(8,4))
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(color_pred)
            axes[1].set_title('Predicted Mask')
            axes[1].axis('off')

            plt.tight_layout()
            plt.show()

            count += 1
        if count >= MAX_IMAGES:
            break
