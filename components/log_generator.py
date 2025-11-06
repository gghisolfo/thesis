import os
import cv2
import pickle
import numpy as np
from copy import deepcopy

# === Config ===
INPUT_MASKS_DIR = "../semantic_segmentation/mini_dataset/predictions"
OUTPUT_PKL_PATH = "../logs/arkanoid_logs/reconstructed_log_no_reference.pkl" #"../logs/arkanoid_logs/reconstructed_log_no_reference.pkl" | "../logs/arkanoid_logs/prova.pkl"


# === Parametri griglia / struttura ===
grid_width, grid_height = 121, 71

# Mappatura class_id nelle maschere -> nomi elementi
CLASS_TO_ELEMENT = {
    1: 'ball',
    2: 'paddle_left',
    3: 'paddle_center',
    4: 'paddle_right',
    5: 'wall_left',
    6: 'wall_right',
    7: 'wall_top',
    8: 'wall_bottom'
}

# === Funzioni di supporto ===

def extract_positions_from_mask(mask, class_id):
    """Estrae la posizione e dimensioni di TUTTI gli oggetti di una certa classe nella maschera"""
    if class_id not in np.unique(mask):
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask == class_id).astype(np.uint8), connectivity=8
    )

    positions = []
    for i in range(1, num_labels):  # salta lo sfondo (label 0)
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        positions.append((int(cx), int(cy), int(w), int(h)))

    return positions

def extract_unique_position_from_mask(mask, class_id):
    """Estrae la posizione centrale di un oggetto dalla maschera"""
    if class_id not in np.unique(mask):
        return None
    
    # Crea un'immagine binaria dove i pixel del class_id sono 1 e gli altri 0, quindi trova componenti connesse.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask == class_id).astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 1:
        return None
    
    # Se più componenti → prendi la più grande
    largest_idx = 1
    if num_labels > 2:
        largest_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_idx = i
    
    x, y, w, h, area = stats[largest_idx]
    cx, cy = centroids[largest_idx]
    return int(cx), int(cy), int(w), int(h)



def create_element(class_id, position):
    """Crea un dizionario per un elemento del gioco"""
    cx, cy, w, h = position
    element_name = CLASS_TO_ELEMENT.get(class_id, f'brick')#{class_id - 9}

    shape_x = max(1, w // 2)
    shape_y = max(1, h // 2)
    
    element = {
        'id': class_id,
        'pos_x': cx,
        'pos_y': cy,
        'shape_x': shape_x,
        'shape_y': shape_y,
        'hitbox_tl_x': cx - shape_x,
        'hitbox_tl_y': cy - shape_y,
        'hitbox_br_x': cx + shape_x,
        'hitbox_br_y': cy + shape_y,
        'color_r': 255,
        'color_g': 255,
        'color_b': 255,
        'color_state': 0,
        'never_hit': True,
        'existence': True,
    }
    return element_name, element


def reconstruct_log_from_masks():
    """Crea un log PKL a partire solo dalle maschere di segmentazione"""
    print(f"📂 Lettura maschere da: {INPUT_MASKS_DIR}")
    mask_files = sorted([
        f for f in os.listdir(INPUT_MASKS_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    reconstructed_frames = []
    total_masks = len(mask_files)
    print(f"Trovate {total_masks} maschere")

    for frame_id, filename in enumerate(mask_files):
        mask_path = os.path.join(INPUT_MASKS_DIR, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print(f"⚠️ Impossibile leggere la maschera: {mask_path}")
            continue

        frame_data = {
            'frame_id': frame_id,
            'elements': {},
            'commands': [],        # aggiunto
            'events': [],          # aggiunto
            'global_state': {},    # aggiunto
        }

        # Oggetti base (classi note) quii
        for class_id, name in CLASS_TO_ELEMENT.items():
            pos = extract_unique_position_from_mask(mask, class_id)
            if pos is not None:
                el_name, el_data = create_element(class_id, pos)
                frame_data['elements'][el_name] = el_data
        
        # Bricks (class_id >= 9)
        brick_classes = [cid for cid in np.unique(mask) if cid >= 9]
        brick_counter = 0

        for cid in brick_classes:
            positions = extract_positions_from_mask(mask, cid)
            for pos in positions:
                el_name, el_data = create_element(cid, pos)
                # dai a ciascun brick un nome unico
                el_name = f"{el_name}_{brick_counter}" 
                frame_data['elements'][el_name] = el_data
                brick_counter += 1
            # else:
            #     print("No clear position for this element")
            #     print(cid)
                
        reconstructed_frames.append(frame_data)

        if frame_id % 100 == 0:
            print(f"🧩 Processato frame {frame_id}/{total_masks}")
    
    # Salvataggio finale
    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(reconstructed_frames, f)
    
    print(f"✅ Log ricostruito: {OUTPUT_PKL_PATH}")
    print(f"Totale frame salvati: {len(reconstructed_frames)}")

    # Debug del primo frame
    if reconstructed_frames:
        first = reconstructed_frames[0]
        print("\n🔍 Elementi nel primo frame:")
        for name in first['elements']:
            print(f"  - {name}")

# === Main ===
if __name__ == "__main__":
    reconstruct_log_from_masks()
