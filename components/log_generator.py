import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

# === Config ===
INPUT_MASKS_DIR = "../semantic_segmentation/fake_prediction"
OUTPUT_PKL_PATH = "../logs/arkanoid_logs/reconstructed_log.pkl"

# Etichette originali
LABELS = {
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

# Colori fissi
CLASS_COLORS = {
    "environment": (0, 0, 0),
    "ball": (255, 0, 0),
    "paddle_left": (0, 0, 255),
    "paddle_center": (0, 100, 255),
    "paddle_right": (0, 150, 255),
    "wall_left": (0, 255, 0),
    "wall_right": (0, 255, 50),
    "wall_top": (0, 255, 150),
    "wall_bottom": (0, 255, 200),
    "brick": (255, 255, 255)
}

# Mapping compatibile con Patch
DESCRIPTION_MAPPING = {
    "paddle_left": "paddle",
    "paddle_center": "paddle",
    "paddle_right": "paddle",
    "ball": "ball",
    "wall_left": "wall",
    "wall_right": "wall",
    "wall_top": "wall",
    "wall_bottom": "wall",
    "brick": "brick",
    "environment": "environment"
}

# Bounding box
def get_bbox(mask, class_id):
    ys, xs = np.where(mask == class_id)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "hitbox_tl_x": int(xs.min()),
        "hitbox_tl_y": int(ys.min()),
        "hitbox_br_x": int(xs.max()),
        "hitbox_br_y": int(ys.max())
    }

# Ricostruzione frames
frames = []
mask_files = sorted(os.listdir(INPUT_MASKS_DIR))
print(f"🖼️ Numero maschere trovate: {len(mask_files)}")

for frame_id, mask_file in tqdm(enumerate(mask_files), total=len(mask_files)):
    mask_path = os.path.join(INPUT_MASKS_DIR, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    elements = {}

    for class_id, name in LABELS.items():
        bbox = get_bbox(mask, class_id)
        r, g, b = CLASS_COLORS[name]

        if bbox is None:
            # Default per oggetti assenti
            elements[name] = {
                "existence": False,
                "color_r": r,
                "color_g": g,
                "color_b": b,
                "pos_x": -1,
                "pos_y": -1,
                "shape_x": 0,
                "shape_y": 0,
                "hitbox_tl_x": -1,
                "hitbox_tl_y": -1,
                "hitbox_br_x": -1,
                "hitbox_br_y": -1,
                "description": DESCRIPTION_MAPPING[name]
            }
        else:
            width = bbox["hitbox_br_x"] - bbox["hitbox_tl_x"] + 1
            height = bbox["hitbox_br_y"] - bbox["hitbox_tl_y"] + 1
            pos_x = bbox["hitbox_tl_x"] + width // 2
            pos_y = bbox["hitbox_tl_y"] + height // 2

            elements[name] = {
                "existence": True,
                "color_r": r,
                "color_g": g,
                "color_b": b,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "shape_x": width,
                "shape_y": height,
                **bbox,
                "description": DESCRIPTION_MAPPING[name]
            }

    frames.append({
        "frame_id": frame_id,
        "commands": [],
        "elements": elements,
        "events": []
    })

# Salvataggio PKL
os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
with open(OUTPUT_PKL_PATH, "wb") as f:
    pickle.dump(frames, f)

print(f"✅ Log ricostruito e salvato in {OUTPUT_PKL_PATH}")

# Controllo rapido
with open(OUTPUT_PKL_PATH, "rb") as f:
    test = pickle.load(f)
print(f"Frames nel log: {len(test)}")
print(f"Elementi attivi nel primo frame: {[v['description'] for k,v in test[0]['elements'].items() if v['existence']]}")
