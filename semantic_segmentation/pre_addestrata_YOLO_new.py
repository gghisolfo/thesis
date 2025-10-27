from ultralytics import YOLO
import cv2
import os
import numpy as np

# Cartelle
input_folder = './prova_images/'
output_folder_masks = './prova_images/segment_masks/'       # mappa pixel per classe
output_folder_visual = './prova_images/segment_visual/'     # visualizzazione colorata

os.makedirs(output_folder_masks, exist_ok=True)
os.makedirs(output_folder_visual, exist_ok=True)

# Carica modello YOLOv8 pre-addestrato per segmentazione
model = YOLO('yolov8n-seg.pt')  # versione segmentazione

# Colori casuali per visualizzazione
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(model.names), 3), dtype=np.uint8)


image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

# Loop su tutte le immagini nella cartella
for img_name in image_files:
    # if not (filename.endswith('.png') or filename.endswith('.jpg')):
    #     continue
    
    img_path = os.path.join(input_folder, img_name)
    results = model(img_path, conf=0.25)  # inference
    
    # Prendi la prima immagine (batch size=1)
    result = results[0]

    # Estrai maschere segmentazione (come boolean mask)
    masks = result.masks.data.cpu().numpy()  # shape: [num_objects, H, W]
    class_ids = result.masks.cls.cpu().numpy()  # class per ogni maschera

    # Crea mappa pixel per classe
    height, width = masks.shape[1], masks.shape[2]
    class_map = np.zeros((height, width), dtype=np.uint8)

    # Sovrapponi tutte le maschere (attenzione all'ordine)
    for i in range(len(masks)):
        mask = masks[i]
        class_map[mask.astype(bool)] = class_ids[i]  # assegna codice classe

    # Salva mappa pixel per classe
    mask_path = os.path.join(output_folder_masks, img_name)
    cv2.imwrite(mask_path, class_map)

    # Crea visualizzazione colorata
    visual = np.zeros((height, width, 3), dtype=np.uint8)
    for cls_id in range(len(model.names)):
        visual[class_map == cls_id] = colors[cls_id]

    visual_path = os.path.join(output_folder_visual, img_name)
    cv2.imwrite(visual_path, visual)

    print(f"Processata {img_name}")
