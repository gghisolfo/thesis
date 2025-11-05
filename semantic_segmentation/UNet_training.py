import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from UNet import UNet
from deep_labv3_plus import get_deeplabv3plus_model
from segmentation import SegmentationDataset, map_mask, CLASS_COLORS
from EarlyStopping import EarlyStopping
from sklearn.model_selection import train_test_split

# -----------------------
# Config
# -----------------------
USE_DEEPLAB = False
IMAGE_SIZE = (120, 70)
NUM_CLASSES = 10
BATCH_SIZE = 4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHUFFLE = False
SHOW_PLOTS = True
SAVE_MODEL = True
SAVE_PREDICTION = False

images_path = "./dataset/images" # "./dataset/images" | "./dataset_complete/images"
masks_path = "./dataset/masks" # "./dataset/images" | "./dataset_complete/images"

# -----------------------
# Utility
# -----------------------

def denormalize(tensor, mean, std):
    for i in range(len(mean)):
        tensor[i] = tensor[i] * std[i] + mean[i]
    return tensor


def show_image(image_tensor):
    image_tensor = denormalize(image_tensor.clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = image_tensor.permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def show_mask(mask_array, num_classes=10):
    # mask_array: numpy HxW with class ids 0..num_classes-1
    cmap = CLASS_COLORS / 255.0
    colored = cmap[mask_array]
    plt.imshow(colored)
    plt.axis('off')
    plt.show()


def compute_iou(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious


# -----------------------
# Dataloaders
# -----------------------

def get_image_mask_paths(images_dir, masks_dir):
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
    masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
    return images, masks


def make_dataloaders(images_dir=images_path, masks_dir=masks_path, batch_size=BATCH_SIZE, shuffle=SHUFFLE, val_split=0.2):
    all_images, all_masks = get_image_mask_paths(images_dir, masks_dir)
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(all_images, all_masks, test_size=val_split, random_state=42)

    train_dataset = SegmentationDataset(train_imgs, train_masks)
    val_dataset = SegmentationDataset(val_imgs, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# -----------------------
# Model builders
# -----------------------

def build_model(num_classes=NUM_CLASSES, use_deeplab=USE_DEEPLAB, device=DEVICE):
    model = get_deeplabv3plus_model(3, num_classes) if use_deeplab else UNet(3, num_classes)
    return model.to(device)


# -----------------------
# Training / validation loops
# -----------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)['out'] if USE_DEEPLAB else model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def validate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    ious = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out'] if USE_DEEPLAB else model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)
            batch_ious = compute_iou(preds, masks, num_classes)
            ious.append(batch_ious)

    ious = np.array(ious)
    miou_per_class = np.nanmean(ious, axis=0)
    miou = np.nanmean(miou_per_class)
    acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    return val_loss, acc, miou, miou_per_class


# -----------------------
# Visualization helper (on validation set)
# -----------------------

def visualize_predictions(model, loader, max_images=5, device=DEVICE):
    model.eval()
    shown = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out'] if USE_DEEPLAB else model(images)
            preds = torch.argmax(outputs, dim=1)
            for b in range(images.size(0)):
                img = images[b].cpu()
                true_mask = masks[b].cpu().numpy()
                pred_mask = preds[b].cpu().numpy()

                img_vis = denormalize(img.clone(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).permute(1,2,0).numpy()
                color_true = CLASS_COLORS[true_mask]
                color_pred = CLASS_COLORS[pred_mask]

                fig, axes = plt.subplots(1,3,figsize=(12,4))
                axes[0].imshow(img_vis)
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

                shown += 1
                if shown >= max_images:
                    return


# -----------------------
# Main training entrypoint
# -----------------------

def main():
    train_loader, val_loader = make_dataloaders()
    model = build_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=5, delta=0.001, path='best_model.pth')

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, acc, miou, miou_per_class = validate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | mIoU: {miou:.4f}")
        print("mIoU per classe:", miou_per_class)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("⛔ Early stopping triggered — training terminated.")
            break

    if SAVE_MODEL:
        torch.save(model.state_dict(), 'segmentation_model.pth')
        print('Modello salvato come segmentation_model.pth')

    if SHOW_PLOTS:
        visualize_predictions(model, val_loader, max_images=5)


if __name__ == '__main__':
    main()
