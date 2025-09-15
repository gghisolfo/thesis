
# CLASS_COLORS = np.array([
#     [0, 0, 0],        # Classe 0: nero 
#     [255, 0, 0],      # Classe 1: rosso
#     [0, 255, 0],      # Classe 2: verde
#     [0, 0, 255],      # Classe 3: blu
#     [255, 255, 0],    # Classe 4: giallo
#     [255, 0, 255],    # Classe 5: magenta
#     [0, 255, 255],    # Classe 6: ciano
#     [128, 128, 128],  # Classe 7: grigio
#     [255, 165, 0],    # Classe 8: arancione
#     [255, 255, 255],  # Classe 9: bianco
# ], dtype=np.uint8)

# masks_path= "./dataset/masks_color"

    
    # mapping_photo = {
    #     0: 0,
    #     23: 1,
    #     76: 2,
    #     150: 3,
    #     165: 4,
    #     195: 5,
    #     210: 6,
    #     230: 7,
    #     240: 8,
    #     255: 9
    # }
    # mapping_short = {
    # 0: 0,
    # 1: 1,
    # 3: 2,
    # 5: 3,
    # 6: 4,
    # 7: 5,
    # 8: 6,
    # 9: 7
    # }

    COLOR_MAP = np.array([
    [0, 0, 0],         # 0 - background - environment (sfondo) - nero
    [0, 0, 255],       # 1 - ball - BLU
    [255, 0, 0],       # 2 - paddle_left - ROSSO ACCESO
    [200, 0, 0],       # 3 - paddle_center - ROSSO medio
    [150, 0, 0],       # 4 - paddle_right - ROSSO scuro
    [0, 255, 0],       # 5 - wall_left - verde acceso
    [0, 255, 50],      # 6 - wall_right - verde acesso
    [0, 255, 150],     # 7 - wall_top - acquamarina
    [0, 255, 150],     # 8 - wall_bottom - acquamarina
    [255, 255, 255]    # 9 - bricks - bianco
], dtype=np.uint8)

# LABELS = {
#     "environment": 0, #background
#     "ball": 1,
#     "paddle_left": 2, #sbarretta
#     "paddle_center": 3, #sbarretta
#     "paddle_right": 4, #sbarretta
#     "wall_left": 5,
#     "wall_right": 6,
#     "wall_top": 7,
#     "wall_bottom": 8,
#     #"bricks": (9, 34),# Bricks are from 9 to 34 
# }
