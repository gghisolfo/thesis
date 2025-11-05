import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from segmentation import CLASS_COLORS_DIFFERENT, CLASS_COLORS_ORIGINAL



# arkanoid_log_2025_09_15_12_02_45.pkl -> 1069 
# arkanoid_log_2025_04_15_09_35_00.pkl -> 211
# arkanoid_log_2025_07_15_16_30_15.pkl -> 398 solo pallina e bordo


# arkanoid_log_2025_02_07_16_03_00.pkl -> PROVA
# arkanoid_log_2025_04_15_09_34_43.pkl -> 20
# arkanoid_log_2025_07_15_16_29_02.pkl -> 124 test

# === Config ===
INPUT_PKL_PATH = "../logs/arkanoid_logs/arkanoid_log_2025_09_15_12_02_45.pkl"
PAD_START = 0  # numero di partenza
COMPLETE = False   

# === Output directories ===
if COMPLETE: #different images colors
    BASE_DIR = "./dataset_complete"
    PALETTE = CLASS_COLORS_DIFFERENT
else:
    BASE_DIR = "./dataset"
    PALETTE = CLASS_COLORS_ORIGINAL

OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_MASKS_DIR = os.path.join(BASE_DIR, "masks")
OUTPUT_MASKS_COLOR_DIR = os.path.join(BASE_DIR, "masks_color")

os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_COLOR_DIR, exist_ok=True)


# === Label definitions ===
LABELS = {
    "environment": 0,   # background
    "ball": 1,
    "paddle_left": 2,
    "paddle_center": 3,
    "paddle_right": 4,
    "wall_left": 5,
    "wall_right": 6,
    "wall_top": 7,
    "wall_bottom": 8,
    "bricks": (9, 34),  # bricks 9–34
}

# === Utility ===
# === Funzione: disegna una bounding box sulla maschera ===
def draw_bbox(mask, obj, label):
    """Disegna una bounding box su una maschera."""
    x1, y1 = int(obj["hitbox_tl_x"]), int(obj["hitbox_tl_y"])
    x2, y2 = int(obj["hitbox_br_x"]), int(obj["hitbox_br_y"])
    mask[y1:y2 + 1, x1:x2 + 1] = label


# === Load pickle ===
with open(INPUT_PKL_PATH, "rb") as f:
    data = pickle.load(f)

print(f"📦 Totale frame caricati: {len(data)} da {INPUT_PKL_PATH}")

# === Process ===
for i, frame in tqdm(enumerate(data), total=len(data), desc="Generazione frame"):
    h, w = 70, 120  # dimensione frame
    rgb = np.zeros((h, w, 3), dtype=np.uint8) #  immagine RGB
    mask = np.zeros((h, w), dtype=np.uint8) # maschera numerica (1 canale)

    for name, obj in frame.get("elements", {}).items():
        if not obj.get("existence", False):
            continue

        # Disegna sul frame RGB
        r, g, b = obj["color_r"], obj["color_g"], obj["color_b"]

        # Disegna nella maschera numerica
        if name in LABELS:
            label = LABELS[name]
        elif name.startswith("brick_"):
            brick_index = int(name.split("_")[1])
            label = 9 #+ brick_index  # 9–34 (come definito)
        else:
            continue

        draw_bbox(mask, obj, label)

        color_index = label if label < len(PALETTE) else 9
        color = PALETTE[color_index].tolist()
        rgb[obj['hitbox_tl_y']:obj['hitbox_br_y']+1,
            obj['hitbox_tl_x']:obj['hitbox_br_x']+1] = color

    frame_id = i + PAD_START

    # === Salvataggi ===
    img_path = os.path.join(OUTPUT_IMAGES_DIR, f"frame_{frame_id:04d}.png")
    mask_path = os.path.join(OUTPUT_MASKS_DIR, f"frame_{frame_id:04d}.png")
    mask_color_path = os.path.join(OUTPUT_MASKS_COLOR_DIR, f"frame_{frame_id:04d}.png")

    # Immagine RGB
    cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Maschera numerica
    cv2.imwrite(mask_path, mask)

    # Maschera colorata (dipende da COMPLETE)
    mask_clamped = mask.copy()
    mask_clamped[mask_clamped > 9] = 9
    mask_color = PALETTE[mask_clamped]

    mask_color_bgr = cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(mask_color_path, mask_color_bgr)

print(f"✅ Dataset generato con successo in: {BASE_DIR}")