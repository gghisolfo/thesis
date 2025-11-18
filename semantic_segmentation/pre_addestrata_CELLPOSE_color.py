import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from matplotlib import cm
from cellpose import io, models, train, core

# Cartelle dataset
train_mask_dir = "./dataset_cellpose/masks"
train_images_dir = "./dataset_cellpose/images"
test_mask_dir = "./dataset_test/masks"
test_images_dir = "./dataset_test/images"

output_dir = "./dataset_cellpose/predictions"
os.makedirs(output_dir, exist_ok=True)
gpu_mode=True

def generate_color_palette(n_colors=30):
    """
    Genera una palette di n_colors vividi, usando un colormap di matplotlib.
    """
    colormap = cm.get_cmap('tab20', n_colors)  # 'tab20' ha colori distinti
    palette = (colormap(range(n_colors))[:, :3] * 255).astype(np.uint8)
    return palette

palette = generate_color_palette(n_colors=30)  # Palette globale

def prepare_data(img_dir, mask_dir, temp_dir, dataset_name="dataset"):
    """
    Prepara i dati copiando immagini e maschere nella stessa cartella.
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    valid_pairs = []
    for fname in os.listdir(img_dir):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid_pairs.append(fname)
        else:
            print(f"⚠️ [{dataset_name}] Salto {fname}: maschera mancante")

    print(f"✅ [{dataset_name}] Coppie valide: {len(valid_pairs)}")
    if len(valid_pairs) == 0:
        return [], [], []

    # Copia file
    for fname in valid_pairs:
        shutil.copy(os.path.join(img_dir, fname),
                    os.path.join(temp_dir, fname))
        
        base_name = os.path.splitext(fname)[0]
        mask_name = base_name + "_masks.png"
        shutil.copy(os.path.join(mask_dir, fname),
                    os.path.join(temp_dir, mask_name))

    # Carica dati
    images, labels, files = io.load_images_labels(temp_dir, mask_filter='_masks')
    return images, labels, files

def train_custom_model(train_img_dir, train_mask_dir, 
                       test_img_dir=None, test_mask_dir=None,
                       model_type='cyto3', n_epochs=100):
    """
    Fine-tuning di Cellpose con train e test separati.
    """
    temp_train_dir = "./dataset_cellpose_temp/train"
    train_data, train_labels, train_files = prepare_data(
        train_img_dir, train_mask_dir, temp_train_dir, "TRAIN"
    )
    if len(train_data) == 0:
        raise ValueError("❌ Nessun dato di training trovato!")

    test_data, test_labels, test_files = None, None, None
    if test_img_dir and test_mask_dir:
        temp_test_dir = "./dataset_cellpose_temp/test"
        test_data, test_labels, test_files = prepare_data(
            test_img_dir, test_mask_dir, temp_test_dir, "TEST"
        )
    
    os.makedirs('./models', exist_ok=True)

    # Inizializza modello
    model = models.CellposeModel(gpu=gpu_mode, model_type=model_type)
    
    # Converte maschere in 2D
    train_labels_proc = [label[:,:,0] if label.ndim==3 else label for label in train_labels]
    test_labels_proc = [label[:,:,0] if label.ndim==3 else label for label in test_labels] if test_labels else None
    
    # Training
    cpmodel_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels_proc,
        test_data=test_data,
        test_labels=test_labels_proc,
        save_path='./models',
        n_epochs=n_epochs,
        learning_rate=0.1,
        weight_decay=0.0001,
        model_name='custom_cellpose'
    )

    # Cleanup
    if os.path.exists("./dataset_cellpose_temp"):
        shutil.rmtree("./dataset_cellpose_temp")
    
    return cpmodel_path

def test_custom_model(model_path, test_img_dir, output_dir):
    """
    Testa il modello custom su nuove immagini e salva le maschere colorate.
    """
    print("\n" + "="*50)
    print("🧪 TEST DEL MODELLO")
    print("="*50)
    
    model = models.CellposeModel(gpu=gpu_mode, pretrained_model=model_path)
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = [f for f in os.listdir(test_img_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📸 Immagini da testare: {len(test_images)}")

    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Immagine non valida: {img_name}")
            continue
        
        masks, flows, styles = model.eval(img, channels=[0,0], diameter=None)
        
        print(f"Debug: mask max = {masks.max()}, unique labels = {np.unique(masks)}")
        
        h, w = masks.shape
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        unique_labels = np.unique(masks)
        for label in unique_labels:
            if label == 0:
                continue
            mask_color[masks == label] = palette[int(label) % len(palette)]
        
        mask_out = os.path.join(output_dir,
                                os.path.splitext(img_name)[0] + "_prediction.png")
        cv2.imwrite(mask_out, mask_color)
        print(f"✅ {img_name} → {mask_out}")

if __name__ == "__main__":
    epochs = 2
    model_type = 'cyto3'
    
    print("\n" + "="*60)
    print("🎓 FASE 1: TRAINING DEL MODELLO")
    print("="*60)
    
    model_path = train_custom_model(
        train_img_dir=train_images_dir,
        train_mask_dir=train_mask_dir,
        test_img_dir=test_images_dir,
        test_mask_dir=test_mask_dir,
        model_type=model_type,
        n_epochs=epochs
    )
    
    print("\n" + "="*60)
    print("🎓 FASE 2: TEST DEL MODELLO")
    print("="*60)
    
    test_custom_model(
        model_path=model_path,
        test_img_dir=test_images_dir,
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("✅ PROCESSO COMPLETATO!")
    print(f"📦 Modello: {model_path}")
    print(f"📁 Predizioni: {output_dir}")
