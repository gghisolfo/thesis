from cellpose import models
import os, cv2
import numpy as np
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

        mask_out = os.path.join(output_dir, os.path.splitext(img_name)[0] + "_mask.png")
        cv2.imwrite(mask_out, mask.astype(np.uint16))
        print(f"✅ Salvata maschera per {img_name} in {mask_out}")

        plt.figure(figsize=(6,6))
        plt.imshow(img, cmap='gray')
        plt.imshow(mask, cmap='nipy_spectral', alpha=0.5)
        plt.title(f"{img_name} - Maschera sovrapposta")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(10000)
        plt.close()

print("🎉 Segmentazione completata!")


# # ========== FASE 2: PREPARAZIONE DATI PER FINE-TUNING ==========
# # Crea annotazioni usando la GUI di Cellpose:
# # cellpose --dir ./prova_images --verbose --save_png

# # Oppure correggi le predizioni attuali salvandole come _seg.npy
# def save_for_training(img, mask, output_path):
#     """Salva immagine e maschera nel formato richiesto da Cellpose"""
#     base_name = os.path.splitext(output_path)[0]
#     # Salva immagine
#     io.imsave(base_name + '.png', img)
#     # Salva maschera in formato _seg.npy
#     np.save(base_name + '_seg.npy', {'masks': mask})


# # ========== FASE 3: FINE-TUNING ==========
# def train_custom_model(train_dir, model_type='cyto2', n_epochs=100):
#     """
#     Fine-tuning del modello Cellpose
    
#     Args:
#         train_dir: cartella con immagini e maschere (_seg.npy)
#         model_type: modello base da cui partire
#         n_epochs: numero di epoche
#     """
#     from cellpose import train
    
#     # Carica dati di training
#     output = io.load_train_test_data(train_dir, mask_filter='_seg.npy')
#     train_data, train_labels, _, test_data, test_labels, _ = output
    
#     # Inizializza modello
#     model = models.CellposeModel(gpu=False, model_type=model_type)
    
#     # Training
#     new_model_path = model.train(
#         train_data, 
#         train_labels,
#         test_data=test_data,
#         test_labels=test_labels,
#         channels=[0,0],  # grayscale
#         save_path='./models',
#         n_epochs=n_epochs,
#         learning_rate=0.1,
#         weight_decay=0.0001,
#         model_name='custom_cellpose'
#     )
    
#     return new_model_path

# # Esempio di utilizzo:
# # train_custom_model('./annotated_images', model_type='cyto3', n_epochs=100)