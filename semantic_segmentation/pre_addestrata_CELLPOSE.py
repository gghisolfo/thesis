
#Cellpose nativamente non segmenta per “tipi di cellule” o più classi, ma per singoli oggetti.

from cellpose import models
import cv2
import matplotlib.pyplot as plt

# path immagine
img_path = "C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/real_images/frame_0008.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Immagine non trovata: {img_path}")

# inizializza modello (usa GPU se disponibile)
model = models.CellposeModel(gpu=False)

# segmentazione
masks_list, flows_list, styles_list = model.eval([img], diameter=None)
mask = masks_list[0]  # estrai l'array dalla lista

# visualizzazione
plt.figure(figsize=(12,6))

# immagine originale
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Originale")
plt.axis('off')

# immagine con maschera
plt.subplot(1,2,2)
plt.imshow(img, cmap='gray')
plt.imshow(mask, cmap='jet', alpha=0.5)  # sovrappone la maschera colorata
plt.title("Segmentata")
plt.axis('off')

plt.show()
