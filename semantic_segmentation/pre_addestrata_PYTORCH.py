import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# =========================
# CONFIGURAZIONE
# =========================
num_classes = 10  # cambia qui con il numero di classi che vuoi
model_name = 'mobilenet_v2'
image_path = 'C:/Users/user/Documents/UNI/TESI/thesis/semantic_segmentation/real_images/frame_0008.png'

# =========================
# CARICA MODELLO
# =========================
# encoder_weights=None se vuoi partire da zero
model = smp.Unet(
    encoder_name=model_name,
    encoder_weights='imagenet',
    classes=num_classes,
    activation=None
)
model.eval()  # modalità inferenza

# =========================
# PREPROCESSING IMMAGINE
# =========================
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # riduci dimensione se sei su CPU
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open(image_path).convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)  # aggiungi dimensione batch

# =========================
# INFERENZA
# =========================
with torch.no_grad():
    output = model(input_tensor)  # output: [1, num_classes, H, W]
    seg_map = torch.argmax(output, dim=1)[0].cpu().numpy()  # [H, W]


# =========================
# RIDIMENSIONA SEGMENTAZIONE ALLA DIM ORIGINALE
# =========================


colors = np.array([
    [0, 0, 0],        # nero
    [255, 0, 0],      # rosso
    [0, 255, 0],      # verde
    [0, 0, 255],      # blu
    [255, 255, 0],    # giallo
    [0, 255, 255],    # ciano
    [255, 0, 255],    # magenta
    [255, 165, 0],    # arancione
    [128, 0, 128],    # viola
    [192, 192, 192]   # grigio chiaro
], dtype=np.uint8)

# crea immagine RGB
seg_rgb = colors[seg_map]

seg_map_resized = np.array(
    Image.fromarray(seg_map.astype(np.uint8)).resize(img.size, resample=Image.NEAREST)
)

# Applica i colori dopo il ridimensionamento
seg_rgb_resized = colors[seg_map_resized]


# =========================
# VISUALIZZAZIONE
# =========================
plt.figure(figsize=(10,5))

# immagine originale
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Originale")
plt.axis('off')



# mappa segmentata
plt.subplot(1,2,2)
plt.imshow(seg_map)  # puoi cambiare cmap
plt.title(f"Segmentazione ({num_classes} classi)")
plt.axis('off')

plt.show()
