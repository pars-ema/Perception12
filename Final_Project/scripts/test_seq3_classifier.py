import os, glob
import cv2, numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms

RECT_ROOT = "Final_Project/data/34759_final_project_rect"
DET_ROOT  = "Final_Project/results/detection_2d"
SEQ3 = "seq_03"

MODEL_PATH = "Final_Project/models/classifier_resnet18.pth"
OUT_DIR = "Final_Project/results/classifier_seq3_vis"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load model + class map
ckpt = torch.load(MODEL_PATH, map_location=device)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v:k for k,v in class_to_idx.items()}

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

left_dir = os.path.join(RECT_ROOT, SEQ3, "image_02", "data")
det_dir  = os.path.join(DET_ROOT,  SEQ3, "data")

img_files = sorted(glob.glob(os.path.join(left_dir, "*.png")))
det_files = sorted(glob.glob(os.path.join(det_dir, "*_detections.npy")))
if not img_files or not det_files:
    raise RuntimeError("Missing seq_03 rectified images or detections.")

n = min(len(img_files), len(det_files))

def safe_crop(img, box_xyxy):
    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w-1, x2); y2 = min(h-1, y2)
    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return None
    return img[y1:y2, x1:x2]

for i in range(n):
    img = cv2.imread(img_files[i])
    dets = list(np.load(det_files[i], allow_pickle=True))
    base = os.path.basename(img_files[i])

    for d in dets:
        crop = safe_crop(img, d["box_xyxy"])
        if crop is None:
            continue

        pil = torchvision.transforms.functional.to_pil_image(crop)
        x = tf(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(1).item()
            cls_name = idx_to_class[pred]

        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, cls_name, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite(os.path.join(OUT_DIR, base), img)

print("Seq3 classifier predictions saved to:", OUT_DIR)
