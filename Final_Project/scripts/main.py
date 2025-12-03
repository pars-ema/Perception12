import os
import sys
import subprocess
import glob
import cv2


# ======================= WHICH STEPS TO RUN =======================

RUN_CALIBRATION         = True   # calibrated.py
RUN_CAMERA_RECT_CHECK   = True   # camera_rectification.py
RUN_BATCH_RECTIFICATION = True   # rectification.py

RUN_DISPARITY           = True   # disparity.py
RUN_DETECTION_2D        = True   # object_detection2D.py
RUN_TRACKING_3D         = True   # tracking3d.py

RUN_YOLO_TRAINER        = True   # YOLOtrainer.py


MAKE_SEGMENT_VIDEOS     = True   # make the mp4 segments
MAKE_FINAL_VIDEO        = True   # concatenate them into final_demo.mp4


# ======================= PATH CONFIG =======================

PYTHON_EXE  = sys.executable
ROOT        = "Final_Project"
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

CALIBRATION_SCRIPT   = os.path.join(SCRIPTS_DIR, "calibrated.py")
CAMERA_RECT_SCRIPT   = os.path.join(SCRIPTS_DIR, "camera_rectification.py")
RECTIFICATION_SCRIPT = os.path.join(SCRIPTS_DIR, "rectification.py")
DISPARITY_SCRIPT     = os.path.join(SCRIPTS_DIR, "disparity.py")
DETECTION_2D_SCRIPT  = os.path.join(SCRIPTS_DIR, "object_detection2D.py")
TRACKING_3D_SCRIPT   = os.path.join(SCRIPTS_DIR, "tracking3d.py")
YOLO_TRAINER_SCRIPT  = os.path.join(SCRIPTS_DIR, "YOLOtrainer.py")  

TRACKING_3D_ROOT  = os.path.join(ROOT, "results", "tracking_3d")
DETECTION_2D_ROOT = os.path.join(ROOT, "results", "detection_2d")
RECTIFIED_ROOT    = os.path.join(ROOT, "data", "34759_final_project_rect")

# root for disparity/depth visualizations
DISPARITY_ROOT    = os.path.join(ROOT, "results", "disparity_depth")

VIDEO_OUTPUT_DIR  = os.path.join(ROOT, "results", "videos")

# sequences and fps for rectified/detections/tracking/depth
SEQS_FOR_RECT_DET_TRACK = ["seq_01", "seq_02"]
FPS = 10

# Your YOLOtrainer output (sequence-3 visualisations)
YOLO_SEQ3_VIS_DIR = os.path.join(ROOT, "results", "yolo_seq3_vis")


# ======================= HELPERS =======================

def run_script(script_path: str, description: str):
    if not os.path.isfile(script_path):
        print(f"[SKIP] {description}: script not found at {script_path}")
        return
    print(f"\n[RUN] {description} --> {script_path}")
    result = subprocess.run([PYTHON_EXE, script_path])
    if result.returncode != 0:
        print(f"[ERROR] {description} failed with exit code {result.returncode}")
    else:
        print(f"[OK] {description} completed")


def make_video_from_images(image_dir: str, out_path: str, fps: int = 10) -> bool:
    if not os.path.isdir(image_dir):
        print(f"[VIDEO] No directory: {image_dir}")
        return False

    img_files = sorted(
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not img_files:
        print(f"[VIDEO] No images in {image_dir}")
        return False

    first = cv2.imread(img_files[0])
    if first is None:
        print(f"[VIDEO] Could not read first image {img_files[0]}")
        return False

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"[VIDEO] Writing {len(img_files)} frames from {image_dir} to {out_path}")

    for i, f in enumerate(img_files):
        frame = cv2.imread(f)
        if frame is None:
            print(f"[VIDEO] Warning: could not read {f}, skipping.")
            continue
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        if (i + 1) % 50 == 0 or i == len(img_files) - 1:
            print(f"[VIDEO]   {i+1}/{len(img_files)} frames written")

    writer.release()
    print(f"[VIDEO] Saved: {out_path}")
    return True


def concat_videos(video_paths, out_path, fps=None):
    ref_path = None
    for p in video_paths:
        if os.path.isfile(p):
            ref_path = p
            break
    if ref_path is None:
        print("[FINAL VIDEO] No input videos, cannot concatenate.")
        return

    cap_ref = cv2.VideoCapture(ref_path)
    ok, frame_ref = cap_ref.read()
    cap_ref.release()
    if not ok or frame_ref is None:
        print(f"[FINAL VIDEO] Could not read reference {ref_path}")
        return

    h, w = frame_ref.shape[:2]
    if fps is None:
        fps = FPS

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"[FINAL VIDEO] Concatenating into {out_path}")
    for vid in video_paths:
        if not os.path.isfile(vid):
            print(f"[FINAL VIDEO]  Skip missing: {vid}")
            continue
        print(f"[FINAL VIDEO]  Adding {vid}")
        cap = cv2.VideoCapture(vid)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        cap.release()

    writer.release()
    print(f"[FINAL VIDEO] Done. Saved {out_path}")


# ======================= MAIN =======================

def main():
    print("============== DTU Perception Final Project: main.py ==============")

    # 1. Calibration / rectification
    if RUN_CALIBRATION:
        run_script(CALIBRATION_SCRIPT, "Step 1: Stereo calibration (calibrated.py)")
    if RUN_CAMERA_RECT_CHECK:
        run_script(CAMERA_RECT_SCRIPT, "Step 2: Rectification check (camera_rectification.py)")
    if RUN_BATCH_RECTIFICATION:
        run_script(RECTIFICATION_SCRIPT, "Step 3: Batch rectification (rectification.py)")

    # 2. Depth + disparity
    if RUN_DISPARITY:
        run_script(DISPARITY_SCRIPT, "Step 4: Disparity + depth (disparity.py)")

    # 3. 2D detections
    if RUN_DETECTION_2D:
        run_script(DETECTION_2D_SCRIPT, "Step 5: 2D detections (object_detection2D.py)")

    # 4. 3D tracking
    if RUN_TRACKING_3D:
        run_script(TRACKING_3D_SCRIPT, "Step 6: 3D tracking (tracking3d.py)")

    # 5. YOLO training / demo pipeline for seq_03
    if RUN_YOLO_TRAINER:
        run_script(YOLO_TRAINER_SCRIPT, "Step 7: YOLO trainer / demo (YOLOtrainer.py)")

    # 6. Create individual videos
    segment_videos = []
    if MAKE_SEGMENT_VIDEOS:
        # loop over both seq_01 and seq_02
        for seq_id in SEQS_FOR_RECT_DET_TRACK:
            # a) rectified seq
            rect_dir = os.path.join(RECTIFIED_ROOT, seq_id, "image_02", "data")
            rect_video = os.path.join(VIDEO_OUTPUT_DIR, f"rectified_{seq_id}.mp4")
            if make_video_from_images(rect_dir, rect_video, fps=FPS):
                segment_videos.append(rect_video)

            # b) DEPTH / DISPARITY VIS
            depth_vis_dir = os.path.join(DISPARITY_ROOT, seq_id, "depth")
            depth_video = os.path.join(VIDEO_OUTPUT_DIR, f"depth_{seq_id}.mp4")
            if make_video_from_images(depth_vis_dir, depth_video, fps=FPS):
                segment_videos.append(depth_video)

            # c) 2D detections seq
            det_vis_dir = os.path.join(DETECTION_2D_ROOT, seq_id, "visualizations")
            det_video = os.path.join(VIDEO_OUTPUT_DIR, f"detections_{seq_id}.mp4")
            if make_video_from_images(det_vis_dir, det_video, fps=FPS):
                segment_videos.append(det_video)

            # d) 3D tracking seq
            track_vis_dir = os.path.join(TRACKING_3D_ROOT, seq_id, "visualizations")
            track_video = os.path.join(VIDEO_OUTPUT_DIR, f"tracking_{seq_id}.mp4")
            if make_video_from_images(track_vis_dir, track_video, fps=FPS):
                segment_videos.append(track_video)

        # e) YOLOtrainer output on seq_03 
        if os.path.isdir(YOLO_SEQ3_VIS_DIR):
            yolo_video = os.path.join(VIDEO_OUTPUT_DIR, "yolo_seq3_demo.mp4")
            if make_video_from_images(YOLO_SEQ3_VIS_DIR, yolo_video, fps=FPS):
                segment_videos.append(yolo_video)
        else:
            print(f"[INFO] YOLO vis dir not found, skipping: {YOLO_SEQ3_VIS_DIR}")

    # 7. Final concatenated video
    if MAKE_FINAL_VIDEO and segment_videos:
        final_video = os.path.join(VIDEO_OUTPUT_DIR, "final_demo.mp4")
        concat_videos(segment_videos, final_video, fps=FPS)
    elif MAKE_FINAL_VIDEO:
        print("[FINAL VIDEO] No segments created, cannot make final_demo.mp4")

    print("\n============== DONE: main.py finished ==============")


if __name__ == "__main__":
    main()
