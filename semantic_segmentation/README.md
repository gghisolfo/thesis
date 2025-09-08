# **dataset_generation_complete.py**

Il file `.pkl` contiene una lista di **frame**.  
Ogni frame ha un dizionario `elements` con tutti gli oggetti presenti nel frame.

### Per ogni frame:
1. Inizializza due immagini:
   - **`rgb`**: l'immagine a colori che userai per visualizzare il frame.
   - **`mask`**: contiene le etichette semantiche come numeri.

### Per ogni oggetto nel frame:
- Se esiste (`existence == True`):
  - Colora il suo **bounding box** nella `rgb` con il suo **colore originale**.
  - Disegna nella `mask` il valore numerico dell’etichetta secondo la mappa `LABELS`.

- Per i **mattoni ("brick") non mappati**, usa l'etichetta `9` come fallback.

# **training.py**
Allenare un modello di segmentazione semantica (U-Net o DeepLabV3+) per distinguere gli elementi del gioco Arkanoid su immagini generate, assegnando a ciascun pixel un'etichetta semantica.
## **Struttura del Dataset**
- **Input**: Immagini RGB dei frame del gioco (70x120 px)
- **Output**: Maschere semantiche, dove ciascun pixel ha un valore intero (classe) tra 0 e 9.
## **Pipeline**
### **1️⃣ Dataset Creation (`SegmentationDataset`)**
- Carica coppie (immagine, maschera).
- Le maschere originali hanno colori in scala di grigi → vengono **rimappate a classi 0-9** tramite la funzione `map_mask`.
- Output: immagini normalizzate in tensor `[3, H, W]`, maschere `[H, W]` con valori interi.
### **2️⃣ Visualizzazione Utility**
- `show_image()`: visualizza immagini normalizzate.
- `show_mask()`: visualizza maschere con colori distintivi tramite `matplotlib`.
### **3️⃣ Modello**
Puoi scegliere tra:
- `U-Net`
- `DeepLabV3+` (se `USE_DEEPLAB = True`)

Output: Tensor `(Batch, Classes, H, W)`
### **4️⃣ Training**
- Loss: `CrossEntropyLoss`
- Ottimizzatore: `Adam`
- Per ogni batch:
    - Ottieni `output = model(image)`
    - `CrossEntropy` richiede **label** come interi `[H, W]`, non one-hot.
    - Calcola il `loss`.
    - Aggiorna pesi.
### **5️⃣ Validation**
- Misura la `pixel accuracy`: rapporto tra pixel corretti e pixel totali.
- Stampa `train_loss`, `val_loss` e `val_accuracy`.
## **6️⃣ Predizione e Visualizzazione dei Risultati**
Per ogni immagine nel validation set:
1. Mostra:
    - **Input Image**
    - **Ground Truth (Colorata per classe)**
    - **Predizione (Colorata per classe)**

# **U-Net**
La U-Net ha due fasi:
- **Encoder** (contrazione) → riduce progressivamente la dimensione spaziale, aumenta il numero di canali.
  -  ogni step:
      - Le feature vengono raffinate (CBR)
      - La risoluzione dimezzata con MaxPool2d(2)
- **Decoder** (espansione) → aumenta la risoluzione, riporta l'informazione spaziale persa.
  - Usi ConvTranspose2d per raddoppiare la risoluzione.
  - Concatenazione (torch.cat) con le feature dell'encoder corrispondente (skip connection).
  - Applichi un CBR per raffinamento.
**output**: 1x1 Conv riduce i canali da 64 a NUM_CLASSES.