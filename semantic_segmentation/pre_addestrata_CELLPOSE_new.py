import os
import cv2
import matplotlib.pyplot as plt
from cellpose import models
import numpy as np

# cartella di input e output
input_dir = "./prova_images"
output_dir = "./prova_images/predictions"
os.makedirs(output_dir, exist_ok=True)

# inizializza modello
model = models.CellposeModel(gpu=False)

# lista immagini
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)

    # carica immagine in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Immagine non valida, salto: {img_name}")
        continue

    # segmentazione
    masks_list, flows_list, styles_list = model.eval([img], diameter=None)
    mask = masks_list[0]

    # salva maschera
    mask_out = os.path.join(output_dir, os.path.splitext(img_name)[0] + "_mask.png")
    cv2.imwrite(mask_out, mask.astype(np.uint16))
    print(f"✅ Salvata maschera per {img_name} in {mask_out}")

    # --- VISUALIZZAZIONE CON SOVRAPPOSIZIONE ---
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')  # immagine originale
    plt.imshow(mask, cmap='nipy_spectral', alpha=0.5)  # maschera semi-trasparente
    plt.title(f"{img_name} - Maschera sovrapposta")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(10000)
    plt.close()

print("🎉 Segmentazione completata!")
