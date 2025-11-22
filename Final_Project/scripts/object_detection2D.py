import os 
import glob 
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


#=========================== CONFIGURATION ======================================
#  Paths to find the rectified sequences 
RECTIFIED_ROOT_CANDIDATES = [
    "Final_Project/data/34759_final_project_rect",   
    "Final_Project/results/rectified",               
    "Final_Project/data/34759_final_project_raw"  
      ]
OUT_ROOT = "Final_Project/results/detection_2d"
MODEL_NAME="yolov8n.pt"


# Detection classes to filter for (COCO dataset)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 7] 
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to accept a detection


def find_rectified_sequences():
    """Find Sequences Candidates in Root"""

    for root in RECTIFIED_ROOT_CANDIDATES:
        if not os.path.isdir(root):
            continue
        seqs= sorted([d for d in os.listdir(root)if d.startswith("seq_")])
        if not seqs:
            continue

        seq_map={}
        for s in seqs:
            seq_path = os.path.join(root, s)

            lA = os.path.join(seq_path, "image_02_left", "provided")
            rA = os.path.join(seq_path, "image_03_right", "provided")

            lB = os.path.join(seq_path, "image_02", "data")
            rB = os.path.join(seq_path, "image_03", "data")

            # We will use the 'left' images (image_02/data or image_02_left/provided) for 2D detection
            if os.path.isdir(lA):
                seq_map[s] = lA
            elif os.path.isdir(lB):
                seq_map[s] = lB

        if seq_map:
            return seq_map

    return {}


def get_class_name(class_id, model):
    """
    Look up the human-readable class name.
    """
    return model.names.get(class_id, f"Class {class_id}")

def process_sequence_detection(seq_name, img_dir, model):
    """
    Running detection on all images in a sequence directory.
    """
    image_files = sorted(glob.glob(os.path.join(img_dir,"*.png")))
    
    if not image_files:
        print(f"Skipping {seq_name}: no images found in {img_dir}.")
        return

    out_seq   = os.path.join(OUT_ROOT, seq_name)
    out_vis   = os.path.join(out_seq, "visualizations")
    out_data  = os.path.join(out_seq, "data")
    os.makedirs(out_vis, exist_ok=True)
    os.makedirs(out_data, exist_ok=True)

    print(f"[{seq_name}] Processing {len(image_files)} images for detection...")


    for img_path in image_files:
        base_name= os.path.splitext(os.path.basename(img_path))[0]


         # Running detection using the YOLO model
            # Using the image path directly is efficient
        results= model(img_path, conf= CONFIDENCE_THRESHOLD, classes = TARGET_CLASSES, verbose = False) 

        #===== EXTRACT and SAVE RAW DATA
        # 
        # 

        detections=[]
        # Getting the first (and usually only) result object
        res = results[0] 

        # Convert Tensor to numpy for easy in handling
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = box
                class_name = get_class_name(cls, model)
                

                
                detections.append({
                    "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": int(cls),
                    "class_name": class_name,
                    "confidence": float(conf)
                })
                
                
                
                 # Save the raw detection data for later 3D projection 
        np.save(os.path.join(out_data, base_name + "_detections.npy"), np.array(detections, dtype=object))

    # == VISUALIZATION and SAVING

     #Drawing bounding boxes on the original image using OpenCV
    

        img = cv2.imread(img_path)
        if img is None: continue
    
        for det in detections:
            x1, y1, x2,y2 = det["box_xyxy"]
            class_name = det["class_name"]
            conf = det["confidence"]

            #Draw Rectangle
            color = (0,255,0) 
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Put label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save the visualized image
        cv2.imwrite(os.path.join(out_vis, base_name + "_detected.png"), img)

    print(f"[{seq_name}] saved detection results to {out_seq}")



def main():
    print("\n=== STEP 4: 2D OBJECT DETECTION (YOLOv8) ===")

    try:
        # Load the YOLOv8 model (it will auto-download if not present)
        model = YOLO(MODEL_NAME)
        print(f"Loaded YOLOv8 model: {MODEL_NAME}")
    except Exception as e:
        print(f"ERROR: Could not load YOLO model. Ensure 'ultralytics' is installed.")
        print(f"Installation command: pip install ultralytics")
        print(f"Detailed error: {e}")
        return

    # Find input image directories
    seq_map = find_rectified_sequences()
    if not seq_map:
        raise RuntimeError("No rectified sequences found. Check rectified root paths.")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Process all sequences
    for seq, img_dir in seq_map.items():
        process_sequence_detection(seq, img_dir, model)


if __name__ == "__main__":
    main()






