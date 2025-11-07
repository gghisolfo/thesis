import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from SegmentationTools import CLASS_COLORS_ORIGINAL, map_mask, SegmentationDataset

# =========================
# CONFIGURAZIONE
# =========================
num_classes = 10
batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
test_split = 0.2
num_epochs = 3 # 7 | 30
training_mode = "fine_tune"  # "decoder_only" | "fine_tune" | "frozen"
SAVE_MODEL = True
images_dir = "./dataset_complete/images" # "./dataset/images" | "./dataset_complete/images"
masks_dir  = "./dataset_complete/masks"

# =========================
# PAD AUTOMATICO
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

# =========================
# DATA AUGMENTATION
# =========================
def augment(img, mask):
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

# =========================
# TRASFORMAZIONI
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
# PREPARA DATASET E DATALOADER
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
# CREA MODELLO U-NET
# =========================
# "decoder_only" parte da ImageNet, allena decoder.
# "fine_tune" riparte dal tuo unet_decoder.pth.
# "frozen" carica direttamente il modello finale (per inferenza).

if training_mode == "decoder_only":
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None
    ).to(device)

elif training_mode == "fine_tune":
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # non ricaricare da ImageNet
        classes=num_classes,
        activation=None
    ).to(device)

    checkpoint_path = "unet_finetuned.pth"
    assert os.path.exists(checkpoint_path), f"{checkpoint_path} non trovato!"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

elif training_mode == "frozen":
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=num_classes,
        activation=None
    ).to(device)

    checkpoint_path = "unet_finetuned.pth"
    assert os.path.exists(checkpoint_path), f"{checkpoint_path} non trovato!"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

else:
    raise ValueError(f"Training mode '{training_mode}' non valido!")




# =========================
# MODALITÀ TRAINING
# =========================
if training_mode == "decoder_only":
    print("Modalità: alleno SOLO il decoder, encoder congelato.")
    for param in model.encoder.parameters():
        param.requires_grad = False
elif training_mode == "fine_tune":
    print("Modalità: fine-tuning completo.")
    for param in model.parameters():
        param.requires_grad = True
elif training_mode == "frozen":
    print("Modalità: rete congelata.")
    for param in model.parameters():
        param.requires_grad = False
else:
    raise ValueError(f"Training mode '{training_mode}' non valido!")

# =========================
# LOSS E OTTIMIZZATORE
# =========================
criterion = nn.CrossEntropyLoss()

if training_mode == "decoder_only":
    # Alleno SOLO il decoder, encoder congelato
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

elif training_mode == "fine_tune":
    # Alleno encoder + decoder, con LR diversi
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": 1e-5},   # encoder più lento
        {"params": model.decoder.parameters(), "lr": 1e-4},   # decoder più veloce
    ])

elif training_mode == "frozen":
    # Nessun training → niente optimizer
    optimizer = None

else:
    raise ValueError(f"Training mode '{training_mode}' non valido!")


best_loss = float('inf')
patience = 3
no_improve = 0

# =========================
# TRAINING LOOP
# =========================
if training_mode != "frozen":
    for epoch in range(num_epochs):
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

        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")
        val_loss = evaluate_loss(model, test_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break


    if SAVE_MODEL :
        save_path = "unet_finetuned.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Modello salvato come {save_path}")


# # =========================
# # INFERENZA SU UNA IMMAGINE
# # =========================
# model.eval()
# test_img_path = test_images[0]
# img = Image.open(test_img_path).convert("RGB")
# img_tensor = image_transform(img).unsqueeze(0).to(device)

# with torch.no_grad():
#     output = model(img_tensor)
#     seg_map = torch.argmax(output, dim=1)[0].cpu().numpy()

# # Pad inverso per tornare alla dimensione originale
# seg_map_resized = np.array(Image.fromarray(seg_map.astype(np.uint8)).resize(img.size, resample=Image.NEAREST))
# seg_rgb = CLASS_COLORS_ORIGINAL[seg_map_resized]

# # =========================
# # FUNZIONE DI VALUTAZIONE mIoU
# # =========================
# def evaluate_miou_and_visualization(model, dataloader, num_classes, device="cpu", max_samples=3):
#     print("Valutazione mIoU ...")
#     model.eval()
    
#     intersection_per_class = np.zeros(num_classes, dtype=np.float64)
#     union_per_class = np.zeros(num_classes, dtype=np.float64)
    
#     samples_processed = 0
#     with torch.no_grad():
#         for imgs, masks in dataloader:
#             imgs, masks = imgs.to(device), masks.to(device)
#             outputs = model(imgs)
#             preds = torch.argmax(outputs, dim=1)

#             for j in range(imgs.size(0)):
#                 image = imgs[j].cpu().permute(1,2,0).numpy()
#                 image = (image - image.min()) / (image.max() - image.min() + 1e-8)

#                 true_mask = masks[j].cpu().numpy()
#                 pred_mask = preds[j].cpu().numpy()

#                 color_true = CLASS_COLORS_ORIGINAL[true_mask]
#                 color_pred = CLASS_COLORS_ORIGINAL[pred_mask]

#                 fig, axes = plt.subplots(1,3,figsize=(15,5))
#                 axes[0].imshow(image); axes[0].set_title("Input"); axes[0].axis("off")
#                 axes[1].imshow(color_true); axes[1].set_title("Ground Truth"); axes[1].axis("off")
#                 axes[2].imshow(color_pred); axes[2].set_title("Prediction"); axes[2].axis("off")
#                 plt.show()

#                 for c in range(num_classes):
#                     pred_c = (pred_mask == c)
#                     mask_c = (true_mask == c)
#                     intersection_per_class[c] += np.logical_and(pred_c, mask_c).sum()
#                     union_per_class[c] += np.logical_or(pred_c, mask_c).sum()

#                 samples_processed += 1
#                 if max_samples is not None and samples_processed >= max_samples:
#                     break
#             if max_samples is not None and samples_processed >= max_samples:
#                 break

#     iou_per_class = intersection_per_class / (union_per_class + 1e-8)
#     miou = np.mean(iou_per_class[union_per_class>0])
#     return miou, iou_per_class

# # =========================
# # VALUTAZIONE
# # =========================
# miou, iou_classes = evaluate_miou_and_visualization(model, test_loader, num_classes, device)
# print(f"\nTest mIoU: {miou:.4f}")
# for i, val in enumerate(iou_classes):
#     print(f"Classe {i}: IoU = {val:.4f}")
