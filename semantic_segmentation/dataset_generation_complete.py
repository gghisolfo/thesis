import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

# === Config ===
INPUT_PKL_PATH = "../logs/arkanoid_logs/arkanoid_log_2025_04_15_09_35_00.pkl"
OUTPUT_IMAGES_DIR = "./dataset/images"
OUTPUT_MASKS_DIR = "./dataset/masks"
OUTPUT_MASKS_COLOR_DIR = "./dataset/masks_color"
pad = 0  # numero di partenza


# arkanoid_log_2025_04_15_09_35_00.pkl -> 211
# arkanoid_log_2025_04_15_09_34_43.pkl -> 20
# arkanoid_log_2025_07_15_16_29_02.pkl -> 124
# arkanoid_log_2025_07_15_16_30_15.pkl -> 398 solo pallina e bordo


# Etichette semantiche
LABELS = {
    "environment": 0,
    "ball": 1,
    "paddle_left": 2,
    "paddle_center": 3,
    "paddle_right": 4,
    "wall_left": 5,
    "wall_right": 6,
    "wall_top": 7,
    "wall_bottom": 8,
    # Bricks are from 9 to 34
}

# === Colormap per visualizzazione (solo per masks_color) ===
COLOR_MAP = np.array([
    [0, 0, 0],         # 0 - environment (sfondo) - nero
    [0, 0, 255],       # 1 - ball - rosso
    [255, 0, 0],       # 2 - paddle_left - blu
    [200, 0, 0],       # 3 - paddle_center - blu medio
    [150, 0, 0],       # 4 - paddle_right - blu scuro
    [0, 255, 0],       # 5 - wall_left - verde
    [0, 255, 50],      # 6 - wall_right - verde lime
    [0, 255, 150],     # 7 - wall_top - acquamarina
    [0, 255, 150],     # 8 - wall_bottom - acquamarina
    [255, 255, 255]    # 9 - bricks - bianco
], dtype=np.uint8)

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_COLOR_DIR, exist_ok=True)

# === Funzione: disegna una bounding box sulla maschera ===
def draw_bbox(mask, obj, label):
    x1, y1 = obj['hitbox_tl_x'], obj['hitbox_tl_y']
    x2, y2 = obj['hitbox_br_x'], obj['hitbox_br_y']
    # Corretto l'ordine: y viene prima (riga), x dopo (colonna)
    mask[y1:y2+1, x1:x2+1] = label

# === Carica il pkl ===
with open(INPUT_PKL_PATH, "rb") as f:
    data = pickle.load(f)




print(f"Totale frame: {len(data)}")

for i, frame in tqdm(enumerate(data), total=len(data)):
    h, w = 70, 120 #Dimensione frame
    rgb = np.zeros((h, w, 3), dtype=np.uint8) #  immagine RGB
    mask = np.zeros((h, w), dtype=np.uint8) # maschera numerica (1 canale)

    for name, obj in frame["elements"].items():
        if not obj["existence"]:
            continue

        # Disegna nel frame RGB
        r, g, b = obj['color_r'], obj['color_g'], obj['color_b']
        rgb[obj['hitbox_tl_y']:obj['hitbox_br_y']+1,
            obj['hitbox_tl_x']:obj['hitbox_br_x']+1] = [r, g, b]

        # Disegna nella maschera numerica
        if name in LABELS:
            label = LABELS[name]
            draw_bbox(mask, obj, label)
        elif name.startswith("brick"):
            draw_bbox(mask, obj, 9)  # fallback brick

    # === Salvataggi ===
    frame_id = i + pad

    # Immagine RGB
    img_path = os.path.join(OUTPUT_IMAGES_DIR, f"frame_{frame_id:04d}.png")
    cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Maschera numerica (1 canale, valori 0–9)
    mask_path = os.path.join(OUTPUT_MASKS_DIR, f"frame_{frame_id:04d}.png")
    cv2.imwrite(mask_path, mask)

    # Maschera colorata (per debug/visualizzazione)
    mask_color = COLOR_MAP[mask]
    mask_color_path = os.path.join(OUTPUT_MASKS_COLOR_DIR, f"frame_{frame_id:04d}.png")
    cv2.imwrite(mask_color_path, mask_color)

print("✅ Dataset generato con successo!")
