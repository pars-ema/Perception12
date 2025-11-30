import os, glob
import cv2
from ultralytics import YOLO
import torch

# ----------------------------
# Paths and settings
# ----------------------------
RECT_ROOT = "/zhome/e5/7/219270/Perception12/Final_Project/data/34759_final_project_rect"
SEQ3 = "seq_03"
OUT_DIR = "/zhome/e5/7/219270/Perception12/Final_Project/results/yolo_seq3_vis"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.3  # you can adjust to show/hide low-confidence detections

# ----------------------------
# Only 3 COCO classes: car, bicycle, pedestrian
# ----------------------------
COCO_CLASSES = ['person','bicycle','car']  # only these 3
ALLOWED_CLASSES = [0, 1, 2]  # corresponding COCO IDs

# ----------------------------
# Load YOLO model
# ----------------------------
yolo_weights = "/zhome/e5/7/219270/Perception12/Final_Project/yolo11m.pt"
yolo_model = YOLO(yolo_weights)
yolo_model.to(device)

# ----------------------------
# Load images
# ----------------------------
img_dir = os.path.join(RECT_ROOT, SEQ3, "image_02", "data")
if not os.path.exists(img_dir):
    raise RuntimeError(f"Directory does not exist: {img_dir}")

img_files = sorted(
    glob.glob(os.path.join(img_dir, "*.png")) +
    glob.glob(os.path.join(img_dir, "*.jpg"))
)

if not img_files:
    raise RuntimeError(f"No images found in {img_dir}")

print(f"Found {len(img_files)} images. Processing...")

# ----------------------------
# Main detection loop
# ----------------------------
for img_path in img_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue

    base = os.path.basename(img_path)
    print(f"Processing {base} ...")

    # YOLO detection
    results = yolo_model(img)
    dets = results[0].boxes.xyxy.cpu().numpy()  # bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # class IDs

    # Draw only allowed classes
    for i, cls_id in enumerate(labels):
        cls_id = int(cls_id)
        if cls_id not in ALLOWED_CLASSES:
            continue
        if scores[i] < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, dets[i])
        cls_name = COCO_CLASSES[ALLOWED_CLASSES.index(cls_id)]

        color = (0, 255, 0)  # green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cls_name} {scores[i]:.2f}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save visualized image
    out_path = os.path.join(OUT_DIR, base)
    cv2.imwrite(out_path, img)

print(f"YOLO predictions for car, bicycle, pedestrian saved to: {OUT_DIR}")
