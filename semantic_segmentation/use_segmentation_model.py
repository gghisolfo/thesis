import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from u_net import UNet  # O deep_labv3_plus se lo usi
# from deep_labv3_plus import get_deeplabv3plus_model

# Config
IMAGE_FOLDER = "./real_images"  # Cartella con immagini reali
MODEL_PATH = "segmentation_model.pth"
NUM_CLASSES = 10
USE_DEEPLAB = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color map per visualizzare maschera
CLASS_COLORS = np.array([
    [0, 0, 0],        # 0
    [255, 0, 0],      # 1
    [0, 255, 0],      # 2
    [0, 0, 255],      # 3
    [255, 255, 0],    # 4
    [255, 0, 255],    # 5
    [0, 255, 255],    # 6
    [128, 128, 128],  # 7
    [255, 165, 0],    # 8
    [255, 255, 255],  # 9
], dtype=np.uint8)


# Image transform (resize, to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((70, 120)),  # Usa le stesse dimensioni del training
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carica il modello
model = UNet(3, NUM_CLASSES) if not USE_DEEPLAB else get_deeplabv3plus_model(3, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# Funzione per visualizzare immagine + maschera predetta
def show_prediction(image_pil, pred_mask):
    image_np = np.array(image_pil)
    pred_color = CLASS_COLORS[pred_mask]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(pred_color)
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# Inference su tutte le immagini nella cartella
image_files = sorted([
    os.path.join(IMAGE_FOLDER, f)
    for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

for path in image_files:
    image_pil = Image.open(path).convert("RGB")
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

    with torch.no_grad():
        output = model(input_tensor)
        output = output['out'] if USE_DEEPLAB else output
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]

    show_prediction(image_pil.resize((120, 70)), pred)  # Assicura stesso size della maschera
