import os
import cv2
import pickle
import numpy as np
from copy import deepcopy

# === Config ===
INPUT_MASKS_DIR = "../semantic_segmentation/dataset/masks"
OUTPUT_PKL_PATH = "../logs/arkanoid_logs/reconstructed_log_clean.pkl"

# Dimensioni griglia dal gioco originale
grid_width, grid_height = 121, 71

# Mappatura ID classe -> nome (mantenere coerente col gioco)
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

# Colori ESATTI dal gioco originale
CLASS_COLORS = {
    0: [0, 0, 0],       # environment
    1: [255, 0, 0],     # ball
    2: [0, 0, 255],     # paddle_left  
    3: [0, 100, 255],   # paddle_center
    4: [0, 150, 255],   # paddle_right
    5: [0, 255, 50],    # wall_left
    6: [0, 255, 100],   # wall_right
    7: [0, 255, 150],   # wall_top
    8: [0, 255, 150],   # wall_bottom (stesso colore di top nel gioco)
    9: [255, 255, 255]  # bricks
}

def create_game_compliant_element(element_name, class_id, cx=None, cy=None, w=None, h=None):
    """Crea elementi conformi alla struttura del gioco originale"""
    
    if element_name == "environment":
        return {
            'id': 0,
            'pos_x': grid_width // 2,
            'pos_y': grid_height // 2,
            'shape_x': grid_width // 2,
            'shape_y': grid_height // 2,
            'hitbox_tl_x': 0,
            'hitbox_tl_y': 0,
            'hitbox_br_x': grid_width - 1,
            'hitbox_br_y': grid_height - 1,
            'color_r': 0, 'color_g': 0, 'color_b': 0,
            'color_state': 0,
            'never_hit': True,
            'existence': False,  # Sempre False nel gioco
        }
    
    elif element_name == "wall_left":
        return {
            'id': 5,
            'pos_x': 1,
            'pos_y': grid_height // 2,
            'shape_x': 1,
            'shape_y': grid_height // 2,
            'hitbox_tl_x': 0,
            'hitbox_tl_y': 3,
            'hitbox_br_x': 2,
            'hitbox_br_y': grid_height - 4,
            'color_r': 0, 'color_g': 255, 'color_b': 50,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
    
    elif element_name == "wall_right":
        return {
            'id': 6,
            'pos_x': grid_width - 2,  # 119
            'pos_y': grid_height // 2,
            'shape_x': 1,
            'shape_y': grid_height // 2,
            'hitbox_tl_x': grid_width - 3,  # 118
            'hitbox_tl_y': 3,
            'hitbox_br_x': grid_width - 1,  # 120
            'hitbox_br_y': grid_height - 4,
            'color_r': 0, 'color_g': 255, 'color_b': 100,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
    
    elif element_name == "wall_top":
        return {
            'id': 7,
            'pos_x': grid_width // 2,
            'pos_y': 1,
            'shape_x': grid_width // 2,
            'shape_y': 1,
            'hitbox_tl_x': 3,
            'hitbox_tl_y': 0,
            'hitbox_br_x': grid_width - 4,
            'hitbox_br_y': 2,
            'color_r': 0, 'color_g': 255, 'color_b': 150,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
    
    elif element_name == "wall_bottom":
        return {
            'id': 8,
            'pos_x': grid_width // 2,
            'pos_y': grid_height - 2,  # 69
            'shape_x': grid_width // 2,
            'shape_y': 1,
            'hitbox_tl_x': 3,
            'hitbox_tl_y': grid_height - 3,  # 68
            'hitbox_br_x': grid_width - 4,
            'hitbox_br_y': grid_height - 1,  # 70
            'color_r': 0, 'color_g': 255, 'color_b': 150,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
    
    elif element_name == "ball":
        # Se non trovato nelle maschere, usa posizione di default
        if cx is None: cx, cy = 44, 46
        ball_radius = 1
        return {
            'id': 1,
            'pos_x': int(cx),
            'pos_y': int(cy),
            'shape_x': ball_radius,
            'shape_y': ball_radius,
            'hitbox_tl_x': int(cx) - ball_radius,
            'hitbox_tl_y': int(cy) - ball_radius,
            'hitbox_br_x': int(cx) + ball_radius,
            'hitbox_br_y': int(cy) + ball_radius,
            'color_r': 255, 'color_g': 0, 'color_b': 0,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
    
    # Per i paddle, creare tutti e 3 con stessi parametri fisici
    elif "paddle" in element_name:
        if cx is None: cx, cy = 60, 60  # Default paddle position
        paddle_halfwidth, paddle_halfheight = 5, 1
        
        paddle_elements = {}
        
        # paddle_left (id: 2)
        paddle_elements['paddle_left'] = {
            'id': 2,
            'pos_x': int(cx),
            'pos_y': int(cy),
            'shape_x': paddle_halfwidth,
            'shape_y': paddle_halfheight,
            'hitbox_tl_x': int(cx) - paddle_halfwidth,
            'hitbox_tl_y': int(cy) - paddle_halfheight,
            'hitbox_br_x': int(cx) + paddle_halfwidth,
            'hitbox_br_y': int(cy) + paddle_halfheight,
            'color_r': 0, 'color_g': 0, 'color_b': 255,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
        
        # paddle_center (id: 3)
        paddle_elements['paddle_center'] = {
            'id': 3,
            'pos_x': int(cx),
            'pos_y': int(cy),
            'shape_x': paddle_halfwidth,
            'shape_y': paddle_halfheight,
            'hitbox_tl_x': int(cx) - paddle_halfwidth,
            'hitbox_tl_y': int(cy) - paddle_halfheight,
            'hitbox_br_x': int(cx) + paddle_halfwidth,
            'hitbox_br_y': int(cy) + paddle_halfheight,
            'color_r': 0, 'color_g': 100, 'color_b': 255,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
        
        # paddle_right (id: 4)
        paddle_elements['paddle_right'] = {
            'id': 4,
            'pos_x': int(cx),
            'pos_y': int(cy),
            'shape_x': paddle_halfwidth,
            'shape_y': paddle_halfheight,
            'hitbox_tl_x': int(cx) - paddle_halfwidth,
            'hitbox_tl_y': int(cy) - paddle_halfheight,
            'hitbox_br_x': int(cx) + paddle_halfwidth,
            'hitbox_br_y': int(cy) + paddle_halfheight,
            'color_r': 0, 'color_g': 150, 'color_b': 255,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
        
        return paddle_elements
    
    # Per brick generici
    elif element_name.startswith("brick"):
        if cx is None or cy is None:
            return None
        
        return {
            'id': class_id,
            'pos_x': int(cx),
            'pos_y': int(cy),
            'shape_x': int(w // 2) if w else 5,
            'shape_y': int(h // 2) if h else 2,
            'hitbox_tl_x': int(cx - (w // 2)) if w else int(cx - 5),
            'hitbox_tl_y': int(cy - (h // 2)) if h else int(cy - 2),
            'hitbox_br_x': int(cx + (w // 2)) if w else int(cx + 5),
            'hitbox_br_y': int(cy + (h // 2)) if h else int(cy + 2),
            'color_r': 255, 'color_g': 255, 'color_b': 255,
            'color_state': 0,
            'never_hit': True,
            'existence': True,
        }
    
    return None


def reconstruct_log():
    frames = []
    mask_files = sorted(os.listdir(INPUT_MASKS_DIR))
    
    for frame_id, fname in enumerate(mask_files):
        mask_path = os.path.join(INPUT_MASKS_DIR, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"⚠️ Impossibile leggere la maschera {mask_path}")
            continue
        
        elements = {}
        
        # 1. SEMPRE aggiungere environment (obbligatorio nel gioco)
        elements['environment'] = create_game_compliant_element('environment', 0)
        
        # 2. SEMPRE aggiungere pareti con parametri fissi (struttura del gioco)
        elements['wall_left'] = create_game_compliant_element('wall_left', 5)
        elements['wall_right'] = create_game_compliant_element('wall_right', 6)
        elements['wall_top'] = create_game_compliant_element('wall_top', 7)
        elements['wall_bottom'] = create_game_compliant_element('wall_bottom', 8)
        
        # 3. Cerca elementi nelle maschere
        ball_found = False
        paddle_found = False
        paddle_cx, paddle_cy = None, None
        
        for class_id in np.unique(mask):
            if class_id == 0:  # Skip environment
                continue
                
            name = LABELS_INV.get(int(class_id), f"unknown_{class_id}")
            
            # Componenti connesse per trovare il centro
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (mask == class_id).astype(np.uint8), connectivity=8
            )
            
            if num_labels > 1:  # Almeno un oggetto trovato
                # Prendi il più grande (index 1 è il primo oggetto, 0 è background)
                largest_idx = 1
                if num_labels > 2:
                    # Trova il componente più grande
                    largest_area = 0
                    for i in range(1, num_labels):
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area > largest_area:
                            largest_area = area
                            largest_idx = i
                
                x, y, w, h, area = stats[largest_idx]
                cx, cy = centroids[largest_idx]
                
                # Ball
                if class_id == 1:  # ball
                    elements['ball'] = create_game_compliant_element('ball', class_id, cx, cy, w, h)
                    ball_found = True
                
                # Paddle (qualsiasi ID paddle lo trattiamo uguale)
                elif class_id in [2, 3, 4]:  # paddle
                    paddle_cx, paddle_cy = cx, cy
                    paddle_found = True
                
                # Bricks
                elif class_id >= 9:  # bricks
                    brick_name = f"brick_{class_id - 9}"
                    elements[brick_name] = create_game_compliant_element(brick_name, class_id, cx, cy, w, h)
        
        # 4. Aggiungi paddle SOLO se trovato nelle maschere (come nel PKL originale)
        if paddle_found:
            paddle_elements = create_game_compliant_element('paddle_any', 3, paddle_cx, paddle_cy)
            if paddle_elements:
                elements.update(paddle_elements)
        
        # 5. Se ball non trovato nelle maschere, non aggiungerlo (comportamento originale)
        if not ball_found and frame_id == 0:  # Solo nel primo frame se necessario
            elements['ball'] = create_game_compliant_element('ball', 1)
        
        # 6. Crea frame nel formato del gioco
        frame_data = {
            'frame_id': frame_id,
            'commands': [],
            'elements': deepcopy(elements),
            'events': [{'description': 'game_start', 'subject': 0}] if frame_id == 0 else []
        }
        frames.append(frame_data)
    
    # Salva PKL
    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(frames, f)
    
    print(f"✅ PKL ricostruito conforme al gioco con {len(frames)} frame: {OUTPUT_PKL_PATH}")

if __name__ == "__main__":
    reconstruct_log()