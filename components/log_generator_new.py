import os
import cv2
import pickle
import numpy as np
from copy import deepcopy

# === Config ===
INPUT_MASKS_DIR = "../semantic_segmentation/dataset/masks"
OUTPUT_PKL_PATH = "../logs/arkanoid_logs/reconstructed_log_clean.pkl"

# Mappatura ID classe -> nome
LABELS_INV = {
    0: "environment",
    1: "ball",
    2: "paddle_left",
    3: "paddle_center",
    4: "paddle_right",
    5: "wall_left",
    6: "wall_right",
    7: "wall_top",
    8: "wall_bottom",
    9: "brick"
}

# Colori coerenti con il tuo gioco
CLASS_COLORS = np.array([
    [0, 0, 0],       # 0 environment
    [255, 0, 0],     # 1 ball
    [0, 0, 255],     # 2 paddle_left
    [0, 100, 255],   # 3 paddle_center
    [0, 150, 255],   # 4 paddle_right
    [0, 255, 0],     # 5 wall_left
    [0, 255, 50],    # 6 wall_right
    [0, 255, 150],   # 7 wall_top
    [0, 255, 150],   # 8 wall_bottom
    [255, 255, 255]  # 9 bricks
], dtype=np.uint8)

# Oggetti obbligatori in ogni frame
MANDATORY_OBJECTS = {"ball", "paddle_left", "paddle_center", "paddle_right",
                     "wall_left", "wall_right", "wall_top", "wall_bottom"}

# Dimensione minima per considerare un patch valido
MIN_SIZE = 2

# === Funzione per creare un elemento compatibile log originale ===
def make_element(class_id, cx, cy, w, h):
    hitbox_tl_x, hitbox_tl_y = int(cx - w // 2), int(cy - h // 2)
    hitbox_br_x, hitbox_br_y = int(cx + w // 2), int(cy + h // 2)
    color = CLASS_COLORS[class_id]
    return {
        "id": int(class_id),
        "pos_x": int(cx),
        "pos_y": int(cy),
        "shape_x": w // 2,
        "shape_y": h // 2,
        "hitbox_tl_x": hitbox_tl_x,
        "hitbox_tl_y": hitbox_tl_y,
        "hitbox_br_x": hitbox_br_x,
        "hitbox_br_y": hitbox_br_y,
        "color_r": int(color[0]),
        "color_g": int(color[1]),
        "color_b": int(color[2]),
        "color_state": 0,
        "never_hit": True,
        "existence": True,
    }

# === Generazione log PKL pulito ===
frames = []
mask_files = sorted(os.listdir(INPUT_MASKS_DIR))

for frame_id, fname in enumerate(mask_files):
    mask_path = os.path.join(INPUT_MASKS_DIR, fname)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"⚠️ Impossibile leggere la maschera {mask_path}")
        continue

    elements = {}
    seen_unique = set()  # evita duplicati di ball e paddle

    for class_id in np.unique(mask):
        if class_id == 0:
            continue  # salta ambiente

        name = LABELS_INV.get(int(class_id), f"unknown_{class_id}")

        # componenti connesse
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (mask == class_id).astype(np.uint8), connectivity=8
        )

        for i in range(1, num_labels):  # salta etichetta sfondo
            x, y, w, h, _ = stats[i]
            cx, cy = centroids[i]

            # Rimuove patch troppo piccole
            if w < MIN_SIZE or h < MIN_SIZE:
                continue

            # Rimuove patch fuori dai bordi
            if not (0 <= cx < mask.shape[1]) or not (0 <= cy < mask.shape[0]):
                continue

            # Rimuove duplicati per oggetti unici
            if name in {"ball", "paddle_left", "paddle_center", "paddle_right"}:
                if name in seen_unique:
                    continue
                seen_unique.add(name)

            key = name
            if name == "brick":
                key = f"brick_{frame_id}_{i}"

            elements[key] = make_element(class_id, cx, cy, w, h)

    # Aggiungi oggetti obbligatori mancanti
    for mandatory in MANDATORY_OBJECTS:
        if mandatory not in elements:
            if "wall" in mandatory:
                if mandatory == "wall_left":
                    cx, cy, w, h = 5, mask.shape[0]//2, 10, mask.shape[0]
                elif mandatory == "wall_right":
                    cx, cy, w, h = mask.shape[1]-5, mask.shape[0]//2, 10, mask.shape[0]
                elif mandatory == "wall_top":
                    cx, cy, w, h = mask.shape[1]//2, 5, mask.shape[1], 10
                elif mandatory == "wall_bottom":
                    cx, cy, w, h = mask.shape[1]//2, mask.shape[0]-5, mask.shape[1], 10
            elif "paddle" in mandatory:
                cx, cy, w, h = mask.shape[1]//2, mask.shape[0]-30, 30, 10
            elif mandatory == "ball":
                cx, cy, w, h = mask.shape[1]//2, mask.shape[0]-50, 8, 8
            class_id = list(LABELS_INV.keys())[list(LABELS_INV.values()).index(mandatory)]
            elements[mandatory] = make_element(class_id, cx, cy, w, h)

    # Salva frame
    frame_data = {
        "frame_id": frame_id,
        "commands": [],
        "elements": deepcopy(elements),
        "events": [{"description": "game_start", "subject": 0}] if frame_id == 0 else []
    }
    frames.append(frame_data)

# Salva PKL
os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
with open(OUTPUT_PKL_PATH, "wb") as f:
    pickle.dump(frames, f)

print(f"✅ PKL ricostruito pulito con {len(frames)} frame: {OUTPUT_PKL_PATH}")
