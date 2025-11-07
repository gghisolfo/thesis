import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import os
from SegmentationTools import SegmentationDataset

# =========================
# CONFIG
# =========================
num_classes = 10
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
test_split = 0.2
max_epochs_decoder = 15
max_epochs_finetune = 15
early_stop_patience = 3
SAVE_BEST_MODEL = True
images_dir = "./dataset_complete/images"
masks_dir  = "./dataset_complete/masks"

# =========================
# PAD E AUGMENTATION
# =========================
def pad_to_multiple_of_32(img):
    w, h = img.size
    new_h = ((h + 31)//32)*32
    new_w = ((w + 31)//32)*32
    padded = Image.new(img.mode, (new_w, new_h))
    padded.paste(img, (0,0))
    return padded

def pad_mask(mask):
    w, h = mask.size
    new_h = ((h + 31)//32)*32
    new_w = ((w + 31)//32)*32
    padded = Image.new("L", (new_w, new_h))
    padded.paste(mask, (0,0))
    return padded

def augment(img, mask):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

# =========================
# TRANSFORMS
# =========================
image_transform = transforms.Compose([
    transforms.Lambda(pad_to_multiple_of_32),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

mask_transform = transforms.Compose([
    transforms.Lambda(pad_mask),
    transforms.PILToTensor(),
    transforms.Lambda(lambda x: x.squeeze(0).long())
])

# =========================
# DATASET E DATALOADER
# =========================
image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".png")])
mask_files  = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".png")])
assert len(image_files) == len(mask_files), "Numero immagini e maschere non corrisponde!"

total_size = len(image_files)
test_size = int(total_size * test_split)
train_size = total_size - test_size

train_images, test_images = image_files[:train_size], image_files[train_size:]
train_masks,  test_masks  = mask_files[:train_size], mask_files[train_size:]

train_dataset = SegmentationDataset(train_images, train_masks, transform=image_transform, mask_transform=mask_transform)
test_dataset  = SegmentationDataset(test_images, test_masks, transform=image_transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Train: {len(train_dataset)} immagini, Test: {len(test_dataset)} immagini")

# =========================
# MODELLO BASE
# =========================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=num_classes,
    activation=None
).to(device)

criterion = nn.CrossEntropyLoss()

# =========================
# FUNZIONI DI SUPPORTO
# =========================
def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_phase(model, optimizer, max_epochs, phase_name, unfreeze_encoder=False):
    print(f"\n============================")
    print(f"INIZIO FASE: {phase_name}")
    print(f"============================")

    if unfreeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = True
    else:
        for p in model.encoder.parameters():
            p.requires_grad = False

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_loss(model, test_loader)

        print(f"Epoch [{epoch+1}/{max_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            if SAVE_BEST_MODEL:
                torch.save(model.state_dict(), f"best_{phase_name}.pth")
                print(f"✅ Miglior modello salvato (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print("⏹ Early stopping per mancanza di miglioramento.")
                break

# =========================
# FASE 1: TRAIN DECODER
# =========================
optimizer_decoder = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
train_phase(model, optimizer_decoder, max_epochs_decoder, phase_name="decoder_only", unfreeze_encoder=False)

# =========================
# FASE 2: FINE-TUNING COMPLETO
# =========================
optimizer_finetune = torch.optim.Adam([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.decoder.parameters(), "lr": 1e-4},
])
train_phase(model, optimizer_finetune, max_epochs_finetune, phase_name="fine_tune", unfreeze_encoder=True)

print("\n✅ Addestramento completato.")
