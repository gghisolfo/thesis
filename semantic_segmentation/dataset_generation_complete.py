import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

# === Config ===
INPUT_PKL_PATH = "./arkanoid_atom/logs/arkanoid_logs/arkanoid_log_2025_04_15_09_35_00.pkl"
OUTPUT_IMAGES_DIR = "./arkanoid_atom/semantic_segmentation/output/images"
OUTPUT_MASKS_DIR = "./arkanoid_atom/semantic_segmentation/output/masks"
OUTPUT_MASKS_COLOR_DIR = "./arkanoid_atom/semantic_segmentation/output/masks_color"

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

for i, frame in tqdm(enumerate(data)):
    h, w = 70, 120
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    for name, obj in frame["elements"].items():
        if not obj["existence"]:
            continue

        r = obj['color_r']
        g = obj['color_g']
        b = obj['color_b']

        # Colori nell'immagine RGB
        rgb[obj['hitbox_tl_y']:obj['hitbox_br_y']+1, obj['hitbox_tl_x']:obj['hitbox_br_x']+1] = [r, g, b]

        # Etichetta: se presente nel dizionario
        if name in LABELS:
            label = LABELS[name]
            draw_bbox(mask, obj, label)
        elif name.startswith("brick"):  # Fallback per eventuali brick non mappati
            draw_bbox(mask, obj, 9)

    # Salvataggio immagini
    img_path = os.path.join(OUTPUT_IMAGES_DIR, f"frame_{i:04d}.png")
    mask_color_path = os.path.join(OUTPUT_MASKS_COLOR_DIR, f"frame_{i:04d}.png")

    cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Crea una LUT personalizzata (colori distinti per label)
    color_map = np.array([
        [0, 0, 0],         # 0 - background - black
        [0, 0, 255],       # 1 - ball -  red
        [255, 0, 0],       # 2 - paddle_left - blue
        [200, 0, 0],       # 3 - paddle_center - mid blue
        [150, 0, 0 ],       # 4 - paddle_right - dark blue
        [0, 255, 0],       # 5 - wall_left - green
        [0, 255, 50],      # 6 - wall_right - Verde con sfumatura lime
        [0, 255, 150],     # 7 - wall_top -	Verde acqua (acquamarina)
        [0, 255, 150],     # 8 - wall_bottom - Verde acqua (acquamarina)
        [255, 255, 255]    # 9 - bricks (fallback) - white
    ], dtype=np.uint8)

    mask_color = color_map[mask]
    cv2.imwrite(mask_color_path, mask_color)

print("✅ Dataset generato!")
