import matplotlib.pyplot as plt
from u_net import UNet
from training import SegmentationDataset
from torch.utils.data import DataLoader
import os

NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Ricrea lo stesso modello della fase di training
model = UNet(3, NUM_CLASSES)  # oppure get_deeplabv3plus_model se usi DeepLab
model = model.to(DEVICE)

# 2. Carica i pesi salvati
model.load_state_dict(torch.load("segmentation_model.pth", map_location=DEVICE))
model.eval()  # importante: modalità eval



# 3. Creare DataLoader per il test
test_images_path = "./dataset/test/images"
test_masks_path = "./dataset/test/masks"

test_imgs = sorted([os.path.join(test_images_path, f) for f in os.listdir(test_images_path)])
test_masks = sorted([os.path.join(test_masks_path, f) for f in os.listdir(test_masks_path)])


test_dataset = SegmentationDataset(test_imgs, test_masks)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 4. Eseguire inferenza
all_preds = []

with torch.no_grad():  # disabilita il calcolo dei gradienti
    for images, masks in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)['out'] if USE_DEEPLAB else model(images)
        preds = torch.argmax(outputs, dim=1)  # [B, H, W]
        all_preds.append(preds.cpu().numpy())

# 5. Visualizzare o salvare le predizioni
for i in range(len(all_preds)):
    pred_mask = all_preds[i][0]  # se batch_size=1
    color_mask = CLASS_COLORS[pred_mask]

    plt.imshow(color_mask)
    plt.title("Predizione")
    plt.axis("off")
    plt.show()
