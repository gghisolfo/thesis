# Etichette semantiche
LABELS = {
    "environment": 0, #background
    "ball": 1,
    "paddle_left": 2, #sbarretta
    "paddle_center": 3, #sbarretta
    "paddle_right": 4, #sbarretta
    "wall_left": 5,
    "wall_right": 6,
    "wall_top": 7,
    "wall_bottom": 8,
    #"bricks": (9, 34),# Bricks are from 9 to 34 
}

# === Colormap per visualizzazione (solo per masks_color) ===
COLOR_MAP = np.array([
    [0, 0, 0],         # 0 - background - environment (sfondo) - nero
    [255, 0, 0],       # 1 - ball - ROSSO
    [0, 0, 255],       # 2 - paddle_left - blu pieno
    [0, 100, 255],     # 3 - paddle_center - blu medio-chiaro
    [0, 150, 255],     # 4 - paddle_right - blu tendente al ciano
    [0, 255, 0],       # 5 - wall_left - verde acceso
    [0, 255, 50],      # 6 - wall_right - verde acesso
    [0, 255, 150],     # 7 - wall_top - acquamarina
    [0, 255, 150],     # 8 - wall_bottom - acquamarina
    [255, 255, 255]    # 9 - bricks - bianco
], dtype=np.uint8)