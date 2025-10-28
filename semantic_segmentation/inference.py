import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp
from UNet import UNet
from segmentation import CLASS_COLORS

# ----------------------- Config -----------------------

NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MAX_IMAGES = 25
IMAGES_FOLDER = "./real_images/images"
SAVE_PREDICTIONS = True
OUTPUT_DIR = "./predictions"
SHOW_IMAGES = False
MODEL_TYPE = "unet"  # "unet" | "smp_unet"
MODEL_PATH = "segmentation_model.pth" # "segmentation_model.pth" | "unet_finetuned.pth"

# ----------------------- Utils -----------------------
def pad_to_multiple_of_32(img: Image.Image):
    """Pad immagine per avere dimensioni multiple di 32 (necessario per molti encoder)."""
    w, h = img.size
    new_w = ((w + 31) // 32) * 32
    new_h = ((h + 31) // 32) * 32
    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(img, (0, 0))
    return padded, (w, h)

def unpad(img: np.ndarray, original_size):
    """Ritaglia immagine alle dimensioni originali dopo padding."""
    w, h = original_size
    return img[:h, :w]

# ----------------------- Dataset -----------------------
class InferenceDataset(Dataset):
    def __init__(self, image_folder):
        self.image_paths = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.image_paths)

    # def __getitem__(self, idx):
    #     path = self.image_paths[idx]
    #     img = Image.open(path).convert("RGB")
    #     padded_img, orig_size = pad_to_multiple_of_32(img)
    #     tensor_img = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1).float() / 255.0
    #     return tensor_img, path, orig_size
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        padded_img, orig_size = pad_to_multiple_of_32(img)
        tensor_img = torch.from_numpy(np.array(padded_img)).permute(2, 0, 1).float() / 255.0

        # Ritorna orig_size come torch tensor di due valori
        return tensor_img, path, torch.tensor(orig_size, dtype=torch.int32)


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

    dataset = InferenceDataset(IMAGES_FOLDER)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if SAVE_PREDICTIONS and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    count = 0
    with torch.no_grad():
        for images, paths, orig_sizes in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for b in range(images.size(0)):
                if count >= MAX_IMAGES:
                    return
                img_path = paths[b]
                #orig_size = tuple(orig_sizes[b])
                #orig_size = tuple(map(int, orig_sizes[b]))  # forza a (w,h)
                print("DEBUG orig_sizes[b]:", orig_sizes[b])
                # orig_size = orig_sizes[b]
                # if isinstance(orig_size, torch.Tensor):
                #     orig_size = orig_size.tolist()
                # if isinstance(orig_size, (list, tuple)) and len(orig_size) == 2:
                #     orig_size = tuple(map(int, orig_size))
                # else:
                #     raise ValueError(f"Formato orig_size non valido: {orig_size}")
                orig_size = orig_sizes[b].tolist()
                if len(orig_size) == 2:
                    orig_size = tuple(map(int, orig_size))
                else:
                    raise ValueError(f"Formato orig_size non valido: {orig_size}")


                pred_mask = preds[b]

                # Ripristina dimensioni originali
                pred_mask = unpad(pred_mask, orig_size)

                # Colora la maschera
                color_pred = CLASS_COLORS[pred_mask]

                # Salva output
                if SAVE_PREDICTIONS:
                    # base_name = os.path.splitext(os.path.basename(img_path))[0]
                    # save_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
                    # Image.fromarray(color_pred.astype(np.uint8)).save(save_path)
                    # print(f"Salvato: {save_path}")

                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
                    Image.fromarray(pred_mask.astype(np.uint8)).save(save_path)
                    print(f"Salvato (maschera grezza): {save_path}")

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

if __name__ == "__main__":
    run_inference()
