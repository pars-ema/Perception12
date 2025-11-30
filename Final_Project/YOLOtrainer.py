import os, glob
import cv2
import torch
from ultralytics import YOLO  # YOLOv8

RECT_ROOT = "Final_Project/data/34759_final_project_rect"
SEQ3 = "seq_03"
OUT_DIR = "Final_Project/results/yolo_seq3_vis"
os.makedirs(OUT_DIR, exist_ok=True)

# Load YOLO model (pretrained)
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov11m.pt")  # small YOLOv8 model; can change to yolov8s.pt, yolov8m.pt etc.
yolo_model.to(device)

# Load images
img_dir = os.path.join(RECT_ROOT, SEQ3, "image_02", "data")
img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
if not img_files:
    raise RuntimeError("No images found.")

for img_path in img_files:
    img = cv2.imread(img_path)
    base = os.path.basename(img_path)

    # YOLO detection
    results = yolo_model(img)  # detect objects
    dets = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # confidence
    labels = results[0].boxes.cls.cpu().numpy()  # class IDs

    # Draw detections
    for i, box in enumerate(dets):
        if scores[i] < 0.3:  # confidence threshold
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"ID:{int(labels[i])} {scores[i]:.2f}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save visualization
    cv2.imwrite(os.path.join(OUT_DIR, base), img)

print("YOLO predictions saved to:", OUT_DIR)
