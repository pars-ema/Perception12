# DTU 34759 – Perception for Autonomous Systems

## Final Project – Object Detection, Depth Estimation & 3D Tracking

This project implements the full perception pipeline required for autonomous driving using stereo vision. It follows the tasks described in the final project assignment and works on all three provided sequences.

---

## Project Structure

```
Final_Project/
│
├── data/
│   ├── calibration/                   # Calibration pattern images
│   └── 34759_final_project_raw/       # Raw stereo data (seq_01, seq_02, seq_03)
│
├── models/                            # Trained model weights
│
├── results/
│   ├── detection_2d/                  # 2D detection outputs + visualizations
│   ├── disparity_depth/               # Depth maps for seq_01 and seq_02
│   ├── rectified/                     # Rectified stereo images
│   ├── tracking_3d/                   # 3D tracking results + images
│   ├── videos/                        # All generated MP4 videos
│   └── yolo_seq3_vis/                 # YOLO trainer test outputs on seq_03
│
├── runs/                              # YOLO training runs
│
├── scripts/
│   ├── calibrated.py                  # Stereo calibration
│   ├── camera_rectification.py        # Rectification check
│   ├── disparity.py                   # Disparity map + depth estimation
│   ├── main.py                        # Full pipeline orchestration + video creation
│   ├── make_val_crops.py              # Validation crop generation
│   ├── make_video.py                  # Video generation utilities
│   ├── object_detection2D.py          # YOLO-based 2D detection
│   ├── rectification.py               # Batch rectification of all sequences
│   ├── test_seq3_classifier.py        # Classifier testing on seq_03
│   ├── tracking3d.py                  # 3D multi-object tracking
│   └── trainer.py                     # Model training utilities
│
├── 3DPointCloud.ipynb                 # 3D point cloud visualization notebook
├── yolo11m.pt                         # YOLO model weights
└── YOLOtrainer.py                     # Training classifier for seq_03
```

---

## What the Pipeline Does

The project completes all goals required in the assignment:

### 1. Stereo Calibration
Uses the provided chessboard images to compute stereo camera parameters.

### 2. Rectification
Raw stereo images are rectified so epipolar lines become horizontal.

### 3. Depth Estimation (Disparity → Depth)
Uses stereo matching to compute disparity maps and convert them into depth maps.

### 4. 2D Object Detection
Runs YOLO on the rectified left images to detect:
- Pedestrians
- Cyclists
- Cars

### 5. 3D Multi-Object Tracking
Combines 2D detections + depth to track objects in 3D, even through occlusions.

### 6. YOLO Classifier Training (Sequence 3)
A small YOLO model is trained to classify objects in the unseen seq_03 test set.

### 7. Video Generation
The pipeline generates multiple synchronized videos showing each task:
- Rectified images
- Depth maps
- 2D detections
- 3D tracking
- YOLO seq_03 demo

A final merged video `final_demo.mp4` is created automatically.

---

## Running the Full Pipeline

Simply run:

```bash
python Final_Project/scripts/main.py
```

The main script will automatically:
- Run all processing steps
- Generate visual outputs
- Produce MP4 videos for seq_01, seq_02, and YOLO seq_03
- Merge them into a final demonstration video

---

## Video Outputs

The following videos will be generated inside `results/videos/`:
