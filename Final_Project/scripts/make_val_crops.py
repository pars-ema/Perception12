# Final_Project/scripts/make_val_crops.py
import os, glob
import numpy as np
import cv2

RECT_ROOT = "Final_Project/data/34759_final_project_rect"
DET_ROOT  = "Final_Project/results/detection_2d"
OUT_VAL   = "Final_Project/data/classifier/val"

SEQ_IDS = ["seq_01", "seq_02"]

# Map COCO YOLO ids -> project 3 classes
def map_yolo_to_project_class(yolo_id: int):
    # COCO: 0 person
    if yolo_id == 0:
        return "pedestrian"
    # COCO: 1 bicycle, 3 motorcycle
    if yolo_id in [1, 3]:
        return "cyclist"
    # COCO: 2 car, 5 bus, 7 truck
    if yolo_id in [2, 5, 7]:
        return "car"
    return None

def safe_crop(img, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w-1, x2); y2 = min(h-1, y2)
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    return img[y1:y2, x1:x2]

def process_seq(seq_name):
    left_dir = os.path.join(RECT_ROOT, seq_name, "image_02", "data")
    det_dir  = os.path.join(DET_ROOT,  seq_name, "data")

    img_files = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    det_files = sorted(glob.glob(os.path.join(det_dir, "*_detections.npy")))

    if not img_files or not det_files:
        print(f"[{seq_name}] Missing rectified images or detections. Check paths.")
        return

    n = min(len(img_files), len(det_files))

    # ensure val class folders exist
    for c in ["pedestrian", "cyclist", "car"]:
        os.makedirs(os.path.join(OUT_VAL, c), exist_ok=True)

    saved = {"pedestrian": 0, "cyclist": 0, "car": 0}

    for i in range(n):
        img = cv2.imread(img_files[i])
        dets = list(np.load(det_files[i], allow_pickle=True))

        base = os.path.splitext(os.path.basename(img_files[i]))[0]
        k = 0

        for d in dets:
            cls = map_yolo_to_project_class(int(d["class_id"]))
            if cls is None:
                continue

            crop = safe_crop(img, d["box_xyxy"])
            if crop is None:
                continue

            out_path = os.path.join(OUT_VAL, cls, f"{seq_name}_{base}_{k}.png")
            cv2.imwrite(out_path, crop)
            saved[cls] += 1
            k += 1

    print(f"[{seq_name}] VAL crops saved:", saved)

if __name__ == "__main__":
    print("=== Building validation set from seq_01 & seq_02 ===")
    for s in SEQ_IDS:
        process_seq(s)
    print("Done. Check Final_Project/results/classifier/val/")
