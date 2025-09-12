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
image_size = (256, 256)    # input U-Net
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
test_split = 0.2           # percentuale immagini da tenere per test
num_epochs = 1             # epoche training

# "decoder_only" ➝ aggiorni solo il decoder.
# "fine_tune" ➝ aggiorni tutto.
# "frozen" ➝ nessun parametro aggiornabile → modello solo per inferenza.
training_mode = "decoder_only"   # "decoder_only" | "fine_tune" | "frozen"


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
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("RGB")

        img = img.resize(image_size)
        mask = mask.resize(image_size, resample=Image.NEAREST)

        if self.transform:
            img, mask = self.transform(img, mask)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)

        mask = np.array(mask)
        mask = mask_rgb_to_class(mask)
        mask = torch.from_numpy(mask).long()

        return img, mask


# =========================
# DATA AUGMENTATION SEMPLICE
# =========================
def augment(img, mask):
    print("DATA AUGMENTATION ... ")
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
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
    (255, 255, 0): 4,
    (0, 255, 255): 5,
    (255, 0, 255): 6,
    (255, 165, 0): 7,
    (128, 0, 128): 8,
    (192, 192, 192): 9
}

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
images_dir = "C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/dataset/images"
masks_dir  = "C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/dataset/masks_color"

image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".png")])
mask_files  = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".png")])

assert len(image_files) == len(mask_files), "Numero immagini e maschere non corrisponde!"

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
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=num_classes,
    activation=None
).to(device)

# Gestione modalità di training
if training_mode == "decoder_only":
    print("Modalità: alleno SOLO il decoder, encoder congelato.")
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = True

elif training_mode == "fine_tune":
    print("Modalità: fine-tuning completo (encoder + decoder allenati).")
    for param in model.parameters():
        param.requires_grad = True

elif training_mode == "frozen":
    print("Modalità: rete completamente congelata (solo inferenza).")
    for param in model.parameters():
        param.requires_grad = False

else:
    raise ValueError(f"Training mode '{training_mode}' non valido!")


# =========================
# LOSS E OTTIMIZZATORE
# =========================
criterion = nn.CrossEntropyLoss()
if training_mode != "frozen":
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
else:
    optimizer = None


# =========================
# TRAINING LOOP (salta se frozen)
# =========================
if training_mode != "frozen":
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if masks.dim() == 4:
                masks = masks.squeeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")


# =========================
# INFERENZA SU UNA IMMAGINE
# =========================
model.eval()
test_img_path = test_images[0]

img = Image.open(test_img_path).convert("RGB")
img_resized = img.resize(image_size)
input_tensor = transforms.ToTensor()(img_resized)
input_tensor = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(input_tensor)
input_tensor = input_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    seg_map = torch.argmax(output, dim=1)[0].cpu().numpy()

seg_map_orig = np.array(Image.fromarray(seg_map.astype(np.uint8)).resize(img.size, resample=Image.NEAREST))

colors = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 165, 0],
    [128, 0, 128],
    [192, 192, 192]
], dtype=np.uint8)

# CLASS_COLORS = np.array([
#     [0, 0, 0],        # Classe 0: nero
#     [255, 0, 0],      # Classe 1: rosso
#     [0, 255, 0],      # Classe 2: verde
#     [0, 0, 255],      # Classe 3: blu
#     [255, 255, 0],    # Classe 4: giallo
#     [255, 0, 255],    # Classe 5: magenta
#     [0, 255, 255],    # Classe 6: ciano
#     [128, 128, 128],  # Classe 7: grigio
#     [255, 165, 0],    # Classe 8: arancione
#     [255, 255, 255],  # Classe 9: bianco
# ], dtype=np.uint8)


seg_rgb = colors[seg_map_orig]


# =========================
# VALUTAZIONE mIoU + VISUALIZZAZIONE
# =========================
def evaluate_miou_and_visualization(model, dataloader, num_classes, device="cpu", max_samples=3):
    """
    Visualizza Input / Ground Truth / Prediction e calcola mIoU
    Gestione batch corretta, predizioni e maschere allineate.
    """
    print("Valutazione mIoU ...")
    model.eval()
    
    intersection_per_class = np.zeros(num_classes, dtype=np.float64)
    union_per_class = np.zeros(num_classes, dtype=np.float64)
    
    samples_processed = 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            batch_size = imgs.size(0)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            for j in range(batch_size):
                # Visualizzazione
                image = imgs[j].cpu().permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)

                true_mask = masks[j].cpu().numpy()
                pred_mask = preds[j].cpu().numpy()

                # Mappa colori (es. CLASS_COLORS- colors definito come array NumPy [num_classes, 3])
                color_true = colors[true_mask]
                color_pred = colors[pred_mask]

                # Visualizzazione
                fig, axes = plt.subplots(1, 3, figsize=(15,5))
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

                # Calcolo IoU
                for c in range(num_classes):
                    pred_c = (pred_mask == c)
                    mask_c = (true_mask == c)
                    intersection_per_class[c] += np.logical_and(pred_c, mask_c).sum()
                    union_per_class[c] += np.logical_or(pred_c, mask_c).sum()

                samples_processed += 1
                if max_samples is not None and samples_processed >= max_samples:
                    break
            if max_samples is not None and samples_processed >= max_samples:
                break

    # Calcolo finale mIoU
    iou_per_class = intersection_per_class / (union_per_class + 1e-8)
    valid_classes = union_per_class > 0
    miou = np.mean(iou_per_class[valid_classes])

    return miou, iou_per_class




miou, iou_classes = evaluate_miou_and_visualization(model, test_loader, num_classes, device)
print(f"\nTest mIoU: {miou:.4f}")
for i, val in enumerate(iou_classes):
    print(f"Classe {i}: IoU = {val:.4f}")


# =========================
# SALVATAGGIO MODELLO
# =========================
save_model = False
if save_model:
    torch.save(model.state_dict(), "unet_weights.pth")
    print("Modello salvato in unet_weights.pth")
