import os
import cv2
import pickle
import numpy as np
from copy import deepcopy

# === Config ===
INPUT_MASKS_DIR = "../semantic_segmentation/real_images/masks"#"../semantic_segmentation/dataset/masks"
OUTPUT_PKL_PATH = "../logs/arkanoid_logs/reconstructed_log_clean.pkl"
REFERENCE_PKL_PATH = "../logs/arkanoid_logs/arkanoid_log_2025_02_07_16_03_00.pkl"

# Dimensioni griglia dal gioco originale
grid_width, grid_height = 121, 71

def extract_position_from_mask(mask, class_id):
    """Estrae la posizione centrale di un oggetto dalla maschera"""
    if class_id not in np.unique(mask):
        return None
    
    # Componenti connesse per trovare il centro
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask == class_id).astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 1:  # Solo background
        return None
    
    # Trova il componente più grande
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

def update_element_position(element, new_pos, new_size=None):
    """Aggiorna la posizione di un elemento mantenendo la struttura originale"""
    if new_pos is None:
        return element
    
    cx, cy, w, h = new_pos
    updated_element = deepcopy(element)
    
    # Aggiorna posizione centrale
    updated_element['pos_x'] = cx
    updated_element['pos_y'] = cy
    
    # Aggiorna shape se fornita
    if new_size:
        updated_element['shape_x'] = new_size[0]
        updated_element['shape_y'] = new_size[1]
    
    # Ricalcola hitbox
    shape_x = updated_element['shape_x']
    shape_y = updated_element['shape_y']
    
    updated_element['hitbox_tl_x'] = cx - shape_x
    updated_element['hitbox_tl_y'] = cy - shape_y
    updated_element['hitbox_br_x'] = cx + shape_x
    updated_element['hitbox_br_y'] = cy + shape_y
    
    return updated_element

def reconstruct_log_from_reference():
    """Ricostruisce il log usando il PKL di riferimento come template"""
    
    # 1. Carica il PKL di riferimento
    print(f"Caricando PKL di riferimento: {REFERENCE_PKL_PATH}")
    with open(REFERENCE_PKL_PATH, 'rb') as f:
        reference_frames = pickle.load(f)
    
    # 2. Ottieni lista maschere
    mask_files = sorted(os.listdir(INPUT_MASKS_DIR))
    
    # 3. Limita al numero di maschere disponibili
    max_frames = min(len(reference_frames), len(mask_files))
    print(f"Elaborando {max_frames} frame")
    
    reconstructed_frames = []
    
    # Mappatura class_id nelle maschere -> nomi elementi
    class_to_element = {
        1: 'ball',
        2: 'paddle_left',
        3: 'paddle_center', 
        4: 'paddle_right',
        5: 'wall_left',
        6: 'wall_right',
        7: 'wall_top',
        8: 'wall_bottom'
    }
    
    for frame_id in range(max_frames):
        # Carica maschera per questo frame
        mask_path = os.path.join(INPUT_MASKS_DIR, mask_files[frame_id])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print(f"⚠️ Impossibile leggere la maschera {mask_path}")
            # Usa frame di riferimento senza modifiche
            reconstructed_frames.append(deepcopy(reference_frames[frame_id]))
            continue
        
        # Copia il frame di riferimento
        reference_frame = reference_frames[frame_id]
        new_frame = deepcopy(reference_frame)
        
        # Aggiorna solo gli elementi presenti nel frame di riferimento
        for element_name, element_data in reference_frame['elements'].items():
            
            # Trova la class_id corrispondente
            class_id = None
            for cid, name in class_to_element.items():
                if name == element_name:
                    class_id = cid
                    break
            
            if class_id is None:
                # Elemento non mappato (es. environment), mantieni originale
                continue
                
            # Estrai nuova posizione dalla maschera
            new_position = extract_position_from_mask(mask, class_id)
            
            if new_position is not None:
                # Aggiorna elemento con nuova posizione
                cx, cy, w, h = new_position
                new_frame['elements'][element_name] = update_element_position(
                    element_data, (cx, cy, w, h)
                )
            # Se non trovato nella maschera, mantieni posizione originale
        
        # Gestione speciale per i brick (ID >= 9)
        brick_ids_in_mask = [cid for cid in np.unique(mask) if cid >= 9]
        
        # Rimuovi brick dal frame di riferimento che non sono nelle maschere
        elements_to_remove = []
        for element_name in new_frame['elements']:
            if element_name.startswith('brick_'):
                brick_num = int(element_name.split('_')[1])
                expected_class_id = brick_num + 9
                if expected_class_id not in brick_ids_in_mask:
                    elements_to_remove.append(element_name)
        
        for element_name in elements_to_remove:
            del new_frame['elements'][element_name]
        
        # Aggiungi brick dalle maschere che non sono nel frame di riferimento
        for class_id in brick_ids_in_mask:
            brick_num = class_id - 9
            brick_name = f'brick_{brick_num}'
            
            if brick_name not in new_frame['elements']:
                new_position = extract_position_from_mask(mask, class_id)
                if new_position:
                    cx, cy, w, h = new_position
                    # Crea nuovo brick basato su template
                    new_frame['elements'][brick_name] = {
                        'id': class_id,
                        'pos_x': cx,
                        'pos_y': cy,
                        'shape_x': max(1, w // 2),
                        'shape_y': max(1, h // 2),
                        'hitbox_tl_x': cx - max(1, w // 2),
                        'hitbox_tl_y': cy - max(1, h // 2),
                        'hitbox_br_x': cx + max(1, w // 2),
                        'hitbox_br_y': cy + max(1, h // 2),
                        'color_r': 255,
                        'color_g': 255,
                        'color_b': 255,
                        'color_state': 0,
                        'never_hit': True,
                        'existence': True,
                    }
            else:
                # Aggiorna posizione brick esistente
                new_position = extract_position_from_mask(mask, class_id)
                if new_position:
                    cx, cy, w, h = new_position
                    new_frame['elements'][brick_name] = update_element_position(
                        new_frame['elements'][brick_name], (cx, cy, w, h)
                    )
        
        reconstructed_frames.append(new_frame)
        
        if frame_id % 100 == 0:
            print(f"Processato frame {frame_id}/{max_frames}")
    
    # 4. Salva PKL ricostruito
    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(reconstructed_frames, f)
    
    print(f"✅ PKL ricostruito con {len(reconstructed_frames)} frame: {OUTPUT_PKL_PATH}")
    
    # 5. Verifica risultato
    print("\nVerifica primo frame:")
    print(f"Elementi nel primo frame: {len(reconstructed_frames[0]['elements'])}")
    for name in reconstructed_frames[0]['elements'].keys():
        print(f"  - {name}")

if __name__ == "__main__":
    reconstruct_log_from_reference()