import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Carica immagine
# -------------------------------
img = cv2.imread('./prova_images/frame_0111.png')
if img is None:
    raise ValueError("Immagine non trovata! Controlla il percorso.")
orig = img.copy()

# Otteniamo le dimensioni dell'immagine per calcolare i filtri
img_height, img_width, _ = img.shape
min_area_threshold = 5   # Abbassato per includere il piccolo quadrato (proiettile)
# L'area massima deve essere appena inferiore all'area totale dello schermo (img_height * img_width).
# Usiamo un fattore (es. 0.95) per escludere il grande rettangolo del bordo.
max_area_threshold = int(img_height * img_width * 0.95) 

# -------------------------------
# 2. Scala di grigi
# -------------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------
# 3. Edge detection con Canny
# -------------------------------
edges = cv2.Canny(gray, 5, 50)

# -------------------------------
# 4. Trova contorni
# Usiamo RETR_LIST per trovare tutti gli oggetti.
# -------------------------------
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# -------------------------------
# 5. Maschera binaria vuota
# -------------------------------
mask = np.zeros_like(gray)
detected_rects = 0

# -------------------------------
# 6. Filtra contorni e disegna rettangoli
# MODIFICHE CHIAVE: Filtro sull'area min e max.
# -------------------------------
for cnt in contours:
    area = cv2.contourArea(cnt)
    
    # Filtra il contorno del campo da gioco (troppo grande)
    if area > max_area_threshold:
        continue
        
    # Filtra il rumore e include il piccolo quadrato (area < 20)
    if area < min_area_threshold:
        continue

    # bounding box
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Disegna il rettangolo sull'immagine originale (orig)
    cv2.rectangle(orig, (x, y), (x+w, y+h), (0,255,0), 2)
    
    # Disegna il rettangolo sulla maschera (mask)
    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    detected_rects += 1


# -------------------------------
# 7. Salva risultati
# -------------------------------
cv2.imwrite("rettangoli_trovati_filtrati.png", orig)
cv2.imwrite("mask_binaria_filtrata.png", mask)
print(f"Rettangoli rilevati (filtrati): {detected_rects}")
print("Immagini salvate: rettangoli_trovati_filtrati.png e mask_binaria_filtrata.png")

# -------------------------------
# 8. Mostra risultati con matplotlib
# -------------------------------
orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(orig_rgb)
plt.title("Rettangoli rilevati (filtrati)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Maschera binaria (filtrata)")
plt.axis('off')

plt.show()