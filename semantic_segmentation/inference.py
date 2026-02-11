import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
from UNet import UNet
from SegmentationTools import CLASS_COLORS_ORIGINAL

# ----------------------- Config -----------------------
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MAX_IMAGES = 100
IMAGES_FOLDER = "./dataset_test/images"
MASKS_FOLDER = "./dataset_test/masks"     # <--- nuovo
SAVE_PREDICTIONS = True
OUTPUT_DIR = "./dataset_test/predictions"
SHOW_IMAGES = False
SAVE_VISIBLE_MASK = False
MODEL_TYPE = "unet"  # "unet" | "smp_unet" 
MODEL_PATH = "models/segmentation_model.pth" # per best_fine_tune ->  smp_unet

# ----------------------- Utils -----------------------
def pad_to_multiple_of_32(img: Image.Image):
    w, h = img.size
    new_w = ((w + 31) // 32) * 32
    new_h = ((h + 31) // 32) * 32
    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(img, (0, 0))
    return padded, (w, h)

def unpad(img: np.ndarray, original_size):
    w, h = original_size
    return img[:h, :w]

# ----------------------- Dataset con maschere -----------------------
class InferenceDatasetWithGT(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_paths = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.mask_folder = mask_folder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        padded_img, orig_size = pad_to_multiple_of_32(img)
        tensor_img = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1).float() / 255.0

        # Maschera GT (grayscale)
        mask_name = os.path.basename(path)
        mask_path = os.path.join(self.mask_folder, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        return tensor_img, torch.tensor(mask, dtype=torch.int64), path, torch.tensor(orig_size, dtype=torch.int32)

# ----------------------- Metriche -----------------------
def compute_iou(pred, target, num_classes):
    """Calcola IoU per ogni classe e ritorna media (mIoU)."""
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    if len(ious) == 0:
        return 0.0
    return np.mean(ious)


# ----------------------- Model builder -----------------------
def build_model(model_type=MODEL_TYPE, num_classes=NUM_CLASSES):
    if model_type == "unet":
        model = UNet(3, num_classes)
    elif model_type == "smp_unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Modello {model_type} non supportato")
    return model


# ----------------------- Inference -----------------------
def run_inference():
    print(f"Carico modello {MODEL_TYPE}...")
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = InferenceDatasetWithGT(IMAGES_FOLDER, MASKS_FOLDER)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if SAVE_PREDICTIONS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_miou = []
    count = 0
    i=0

    with torch.no_grad():
        for images, masks, paths, orig_sizes in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.numpy()

            for b in range(images.size(0)):
                if count >= MAX_IMAGES:
                    # print(count)
                    break

                img_path = paths[b]
                orig_size = tuple(map(int, orig_sizes[b].tolist()))
                pred_mask = unpad(preds[b], orig_size)
                gt_mask = unpad(masks[b], orig_size)

                # Calcola IoU per questa immagine
                miou = compute_iou(pred_mask, gt_mask, NUM_CLASSES)
                total_miou.append(miou)

                # Colora maschera e salva
                color_pred = CLASS_COLORS_ORIGINAL[pred_mask]
                if SAVE_PREDICTIONS:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
                    Image.fromarray(pred_mask.astype(np.uint8)).save(save_path)
                    # print(f"Salvato (maschera grezza): {save_path}")
                
                if i<2 and SAVE_VISIBLE_MASK:
                    i=i+1
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask_overlay.png")
                    Image.fromarray(color_pred.astype(np.uint8)).save(save_path)

                
                # Visualizza
                if SHOW_IMAGES:
                    img = Image.open(img_path).convert("RGB")
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    axes[0].imshow(img)
                    axes[0].set_title("Original")
                    axes[0].axis("off")

                    axes[1].imshow(color_pred)
                    axes[1].set_title("Prediction")
                    axes[1].axis("off")

                    plt.tight_layout()
                    plt.show()
                
                count += 1

    # Metrica finale
    mean_miou = np.mean(total_miou)
    print(f"\n✅ Mean IoU (mIoU): {mean_miou:.4f}")

if __name__ == "__main__":
    run_inference()
