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


# Config
USE_DEEPLAB = False
IMAGE_SIZE = (120, 70)
NUM_CLASSES = 10
BATCH_SIZE = 4
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# images_path= "./arkanoid_atom/semantic_segmentation/output/images"
# masks_path= "./arkanoid_atom/semantic_segmentation/output/masks_color"

images_path= "./output/images"
masks_path= "./output/masks_color"




# Mappatura esplicita dei valori della maschera (esempio)
LABEL_VALUES = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225]  # 10 classi

def map_mask(mask):
    mapped_mask = np.zeros_like(mask)
    
    mapping = {
        0: 0,
        23: 1,
        76: 2,
        150: 3,
        165: 4,
        195: 5,
        210: 6,
        230: 7,
        240: 8,
        255: 9
    }
    
    for original_value, class_index in mapping.items():
        mapped_mask[mask == original_value] = class_index

    return mapped_mask



# Ottieni tutte le coppie immagine-maschera
all_images = sorted([os.path.join(images_path, f) for f in os.listdir(images_path)])
all_masks = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path)])

# Dividi in train e val
train_imgs, val_imgs, train_masks, val_masks = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)



class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"🟡 EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.verbose:
            print(f"✅ Validation loss improved — saving model to {self.path}")
        torch.save(model.state_dict(), self.path)


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
            # mask_np = np.array(mask)
            # print("Mask before mapping:", np.unique(mask_np))

            # mask_np = map_mask(mask_np)
            # print("Mask after mapping:", np.unique(mask_np))


            print("Classi uniche prima della mappatura:", np.unique(np.array(mask)))
            print("Valori unici nella maschera originale:", np.unique(mask))

            mask_np = map_mask(np.array(mask))
            print("Classi uniche dopo la mappatura:", np.unique(mask_np))


            Image.fromarray(mask_np.astype(np.uint8)).save(f"check_mask_raw_{idx}.png")
            Image.fromarray((mask_np * 25).astype(np.uint8)).save(f"check_mask_mapped_{idx}.png")

            mask = Image.fromarray(mask_np.astype(np.uint8))

        mask = Image.fromarray(mask_np.astype(np.uint8))

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


train_dataset = SegmentationDataset(train_imgs, train_masks)
val_dataset = SegmentationDataset(val_imgs, val_masks)


SHUFFLE= False #true
SHOW_PLOTS = True


writer = SummaryWriter()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE) 
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

# Define a function to denormalize the image
def denormalize(tensor, mean, std):
    """ Denormalize the tensor back to the [0, 1] range for visualization. """
    for i in range(len(mean)):
        tensor[i] = tensor[i] * std[i] + mean[i]
    return tensor

def show_image(image_tensor):
    """ Display a transformed image using matplotlib. """
    image_tensor = denormalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Denormalize the image
    image = image_tensor.permute(1, 2, 0).numpy()  # Convert to HxWxC format
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()

def show_mask(mask_tensor, num_classes=10):
    """Visualizza la maschera di segmentazione semantica."""
    mask_np = mask_tensor.detach().cpu().numpy()

    # Caso 1: one-hot (C, H, W)
    if mask_np.ndim == 3:
        label_mask = np.argmax(mask_np, axis=0)  # Shape: (H, W)
    # Caso 2: già label map (H, W)
    elif mask_np.ndim == 2:
        label_mask = mask_np
    else:
        raise ValueError(f"Forma maschera non valida: {mask_np.shape}")

    # Visualizzazione
    plt.imshow(label_mask, cmap='tab20', vmin=0, vmax=num_classes - 1)
    plt.colorbar()
    plt.axis('off')
    plt.title("Mask")
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
            iou = float('nan')  # Ignora classi assenti
        else:
            iou = intersection / union
        ious.append(iou)
    return ious



for i in range(1):
    image, mask = train_dataset[i]
    

    if SHOW_PLOTS:
        show_image(image)
        show_mask(mask)


# Modello
model = get_deeplabv3plus_model(3, NUM_CLASSES) if USE_DEEPLAB else UNet(3, NUM_CLASSES)
model = model.to(DEVICE)

# Loss e ottimizzatore
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

early_stopping = EarlyStopping(patience=5, delta=0.001, path="best_model.pth")



# Training
for epoch in range(EPOCHS):

    print(f"Epoch {epoch+1}/{EPOCHS}")
    
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        
        print("         -> evaluation image")

        images, masks = images.to(DEVICE), masks.to(DEVICE)

        if USE_DEEPLAB:
            outputs = model(images)['out']  # Estratto qui
            loss = criterion(outputs, masks)  # Ora 'outputs' è un tensor
        else:
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
    ious = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            if USE_DEEPLAB:
                outputs = model(images)['out']  # Estratto qui
            else:
                outputs = model(images)

            loss = criterion(outputs, masks)

            #loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)


            # Accumulate IoU
            batch_ious = compute_iou(preds, masks, NUM_CLASSES)
            ious.append(batch_ious)

    #print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}  ")#| mIoU: {miou:.4f}
    #print("mIoU per classe:", miou_per_class)

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("⛔ Early stopping triggered — training terminated.")
        break

               

    # Compute mean IoU over all batches
    ious = np.array(ious)
    miou_per_class = np.nanmean(ious, axis=0)
    miou = np.nanmean(miou_per_class)

    acc = correct_pixels / total_pixels
    # dentro l’epoch
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", acc, epoch)
    writer.add_scalar("mIoU/val", miou, epoch)

    writer.close()


    # print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")


    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | mIoU: {miou:.4f}")
    print("mIoU per classe:", miou_per_class)



torch.save(model.state_dict(), "segmentation_model.pth")
print("Modello salvato come segmentation_model.pth")

    




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
#         image = images[j].cpu().permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
#         true_mask = masks[j].cpu().numpy()
#         pred_mask = preds[j].cpu().numpy()

#         color_true = CLASS_COLORS[true_mask]
#         color_pred = CLASS_COLORS[pred_mask]

#         # Visualizza
#         fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#         axes[0].imshow(image)
#         axes[0].set_title("Input Image")
#         axes[0].axis("off")

#         axes[1].imshow(color_true)
#         axes[1].set_title("Ground Truth")
#         axes[1].axis("off")

#         axes[2].imshow(color_pred)
#         axes[2].set_title("Prediction")
#         axes[2].axis("off")

#         plt.tight_layout()
#         plt.show()

#     # Per mostrare solo il primo batch, rompi qui
#     break

for i, (images, masks) in enumerate(val_loader):
    images, masks = images.to(DEVICE), masks.to(DEVICE)

    outputs = model(images)['out'] if USE_DEEPLAB else model(images)
    preds = torch.argmax(outputs, dim=1)

    for j in range(images.size(0)):
        image = images[j].cpu().permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        true_mask = masks[j].cpu().numpy()
        pred_mask = preds[j].cpu().numpy()

        color_true = CLASS_COLORS[true_mask]
        color_pred = CLASS_COLORS[pred_mask]

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

    #break  # Solo il primo batch
