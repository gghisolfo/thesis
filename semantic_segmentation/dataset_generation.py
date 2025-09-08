import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

# === Config ===
INPUT_PKL_PATH = "arkanoid_atom/logs/arkanoid_logs/arkanoid_log_2025_04_15_09_35_00.pkl"
OUTPUT_IMAGES_DIR = "./output/images"
OUTPUT_MASKS_DIR = "./output/masks"
OUTPUT_MASKS_COLOR_DIR = "./output/masks_color"

# Etichette semantiche
LABELS = {
    "ball": 1,
    "paddle_center": 2,
    "brick": 3,
    "wall": 4,
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
    # Creiamo un'immagine vuota (completa di 3 canali) per il frame RGB
    h, w = 70, 120  # Impostiamo le dimensioni fisse (usando quelle dell'esempio nel dizionario)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)  # Un'immagine vuota con 3 canali RGB

    # Creiamo una maschera vuota
    mask = np.zeros((h, w), dtype=np.uint8)

    # Iteriamo su tutti gli oggetti nel frame
    for name, obj in frame["elements"].items():
        if not obj["existence"]:
            continue
        
        # Estraiamo i colori dall'oggetto
        r = obj['color_r']
        g = obj['color_g']
        b = obj['color_b']

        # Assegniamo i pixel RGB all'immagine (occhio all'ordine y,x)
        rgb[obj['hitbox_tl_y']:obj['hitbox_br_y']+1, obj['hitbox_tl_x']:obj['hitbox_br_x']+1] = [r, g, b]

        # Disegniamo la bounding box sulla maschera
        if name.startswith("brick"):
            draw_bbox(mask, obj, LABELS["brick"])
        elif name.startswith("wall"):
            draw_bbox(mask, obj, LABELS["wall"])
        elif name == "ball":
            draw_bbox(mask, obj, LABELS["ball"])
        elif name == "paddle_center":
            draw_bbox(mask, obj, LABELS["paddle_center"])

    # Salva frame e maschera
    img_path = os.path.join(OUTPUT_IMAGES_DIR, f"frame_{i:04d}.png")
    #mask_path = os.path.join(OUTPUT_MASKS_DIR, f"frame_{i:04d}.png")
    mask_color_path = os.path.join(OUTPUT_MASKS_COLOR_DIR, f"frame_{i:04d}.png")

    cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Salva la maschera in scala di grigi
    #cv2.imwrite(mask_path, mask)

    # Applica una colormap per rendere la maschera colorata
    mask_color = cv2.applyColorMap(mask * 60, cv2.COLORMAP_JET)
    cv2.imwrite(mask_color_path, mask_color)

print("✅ Dataset generato!")
