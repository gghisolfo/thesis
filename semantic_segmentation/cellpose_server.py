from cellpose import models
import os, cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non interattivo
import matplotlib.pyplot as plt

input_dir = "./mini_dataset"
output_dir = "./mini_dataset/predictions"
os.makedirs(output_dir, exist_ok=True)

model_types = ['cyto2', 'cyto3', 'nuclei', 'tissuenet_cp3']

for model_type in model_types:
    print(f"\n🔍 Testando modello: {model_type}")
    model = models.CellposeModel(gpu=False, model_type=model_type)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"⚠️ Immagine non valida, salto: {img_name}")
            continue

        masks, flows, styles = model.eval(img, channels=[0,0], diameter=None)
        mask = masks

        # Salva maschera binaria
        mask_out = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask_{model_type}.png")
        cv2.imwrite(mask_out, mask.astype(np.uint16))
        print(f"✅ Salvata maschera per {img_name} in {mask_out}")

        # Salva immagine overlay
        overlay_out = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_overlay_{model_type}.png")
        plt.figure(figsize=(6,6))
        plt.imshow(img, cmap='gray')
        plt.imshow(mask, cmap='nipy_spectral', alpha=0.5)
        plt.title(f"{img_name} - Maschera sovrapposta")
        plt.axis('off')
        plt.savefig(overlay_out)
        plt.close()
        print(f"🖼️ Salvata immagine overlay in {overlay_out}")

print("🎉 Segmentazione completata!")
