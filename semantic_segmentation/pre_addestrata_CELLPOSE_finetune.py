import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from cellpose import io, models, train, core

# Cartelle separate per train e test
train_mask_dir = "./dataset_cellpose/masks"
train_images_dir = "./dataset_cellpose/images"
test_mask_dir = "./dataset_test/masks"
test_images_dir = "./dataset_test/images"

output_dir = "./dataset_cellpose/predictions"
os.makedirs(output_dir, exist_ok=True)


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
        print(f"⚠️ [{dataset_name}] Nessuna coppia trovata")
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
    images = io.load_images_labels(temp_dir, mask_filter='_masks')[0]
    labels = io.load_images_labels(temp_dir, mask_filter='_masks')[1]
    files = io.load_images_labels(temp_dir, mask_filter='_masks')[2]
    
    return images, labels, files


def train_custom_model(train_img_dir, train_mask_dir, 
                       test_img_dir=None, test_mask_dir=None,
                       model_type='cyto3', n_epochs=100):
    """
    Fine-tuning di Cellpose 4.0+ con train e test separati.
    """
    
    # Prepara training set
    print("\n" + "="*50)
    print("📦 PREPARAZIONE TRAINING SET")
    print("="*50)
    temp_train_dir = "./dataset_cellpose_temp/train"
    train_data, train_labels, train_files = prepare_data(
        train_img_dir, train_mask_dir, temp_train_dir, "TRAIN"
    )
    
    if len(train_data) == 0:
        raise ValueError("❌ Nessun dato di training trovato!")
    
    # Prepara test set (opzionale)
    test_data, test_labels, test_files = None, None, None
    if test_img_dir and test_mask_dir:
        print("\n" + "="*50)
        print("📦 PREPARAZIONE TEST SET")
        print("="*50)
        temp_test_dir = "./dataset_cellpose_temp/test"
        test_data, test_labels, test_files = prepare_data(
            test_img_dir, test_mask_dir, temp_test_dir, "TEST"
        )
    
    # Summary
    print("\n" + "="*50)
    print("📊 RIEPILOGO DATASET")
    print("="*50)
    print(f"🎯 Training samples: {len(train_data)}")
    if test_data:
        print(f"🎯 Test samples: {len(test_data)}")
    else:
        print("🎯 Test samples: Nessuno (verrà usato il 10% del training)")
    
    # Crea cartella per i modelli
    os.makedirs('./models', exist_ok=True)
    
    # Training con API corretta per Cellpose 4.0+
    print("\n" + "="*50)
    print("🚀 INIZIO TRAINING")
    print("="*50)
    
    # Inizializza il modello
    model = models.CellposeModel(gpu=False, model_type=model_type)
    
    # Converti le maschere nel formato corretto se necessario
    train_labels_proc = []
    for label in train_labels:
        if label.ndim == 2:
            train_labels_proc.append(label)
        else:
            train_labels_proc.append(label[:,:,0] if label.shape[2] > 0 else label)
    
    if test_labels:
        test_labels_proc = []
        for label in test_labels:
            if label.ndim == 2:
                test_labels_proc.append(label)
            else:
                test_labels_proc.append(label[:,:,0] if label.shape[2] > 0 else label)
    else:
        test_labels_proc = None
    
    # Training usando il metodo corretto
    # cpmodel_path = model.train(
    #     train_data=train_data,
    #     train_labels=train_labels_proc,
    #     train_files=train_files,
    #     test_data=test_data,
    #     test_labels=test_labels_proc,
    #     test_files=test_files,
    #     save_path='./models',
    #     save_every=10,
    #     n_epochs=n_epochs,
    #     learning_rate=0.1,
    #     weight_decay=0.0001,
    #     model_name='custom_cellpose'
    # )
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
    model_name='custom_cellpose')


    print(f"\n✅ Modello salvato in: {cpmodel_path}")
    
    # Cleanup cartelle temporanee
    if os.path.exists("./dataset_cellpose_temp"):
        shutil.rmtree("./dataset_cellpose_temp")
    
    return cpmodel_path


def test_custom_model(model_path, test_img_dir, output_dir):
    """
    Testa il modello custom su nuove immagini.
    """
    print("\n" + "="*50)
    print("🧪 TEST DEL MODELLO")
    print("="*50)
    
    # Carica modello custom
    model = models.CellposeModel(gpu=False, pretrained_model=model_path)
    
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
        
        # Predizione
        masks, flows, styles = model.eval(img, channels=[0,0], diameter=None)
        
        # Salva maschera colorata
        if masks.max() > 0:
            mask_color = plt.get_cmap("nipy_spectral")(masks.astype(np.float32) / masks.max())
            mask_color = (mask_color[:, :, :3] * 255).astype(np.uint8)
        else:
            mask_color = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
            
        mask_out = os.path.join(output_dir, 
                               os.path.splitext(img_name)[0] + "_prediction.png")
        cv2.imwrite(mask_out, cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR))
        
        # Salva overlay
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_rgb = cv2.cvtColor(img_norm.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = img_rgb.copy()
        overlay_mask = (masks > 0).astype(np.uint8) * 255
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], overlay_mask)
        overlay_out = os.path.join(output_dir, 
                                   os.path.splitext(img_name)[0] + "_overlay.png")
        cv2.imwrite(overlay_out, overlay)
        
        print(f"✅ {img_name} → {mask_out}")
    
    print(f"\n🎉 Predizioni salvate in: {output_dir}")


if __name__ == "__main__":
    # Parametri
    epochs = 2 # Ridotto per evitare overfit con solo 20 immagini
    model_type = 'cyto3'
    
    # 1. TRAINING
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
    
    # 2. TEST
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
    print("="*60)
    print(f"📦 Modello: {model_path}")
    print(f"📁 Predizioni: {output_dir}")