from ultralytics import YOLO
import cv2

# Carica modello pre-addestrato
model = YOLO('yolov8n.pt')  # Pre-trained on COCO dataset

# Inference sulla tua immagine
results = model('./prova_images/frame_0111.png', conf=0.25)

# Visualizza risultati
results[0].show()

# Estrai detections
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    print(f"Oggetto: {model.names[class_id]}, Pos: ({x1},{y1},{x2},{y2})")