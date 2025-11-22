import os
import glob
import numpy as np
import cv2

# ======================= CONFIG =======================
# Path to the calibration file generated in Step 1
CALIB_NPZ = "Final_Project/models/stereo_calibration.npz"

# Root folder containing the raw, unrectified sequence data
# We assume this structure: RAW_ROOT/seq_xx/image_02/data/*.png (Left)
# and RAW_ROOT/seq_xx/image_03/data/*.png (Right)
RAW_ROOT = "Final_Project/data/34759_final_project_raw" 

# Output folder for the new, perfectly rectified images.
# This folder is what the disparity/depth script (Step 3) will use.
OUT_RECTIFIED_ROOT = "Final_Project/results/rectified"
# ======================================================

def load_calibration_maps(calib_path, img_shape):
    """
    Loads all required matrices from the NPZ file and computes the
    undistort and rectify mapping matrices for both cameras.
    """
    data = np.load(calib_path, allow_pickle=True)
    
    # Intrinsic and Extrinsic Matrices
    mtxL = data["mtxL"]
    distL = data["distL"]
    mtxR = data["mtxR"]
    distR = data["distR"]
    
    # Rectification Matrices
    R1 = data["R1"]
    P1 = data["P1"]
    R2 = data["R2"]
    P2 = data["P2"]

    # Compute the rectification maps
    # map1x, map1y: maps for the Left camera (R1/P1)
    map1x, map1y = cv2.initUndistortRectifyMap(
        mtxL, distL, R1, P1, img_shape, cv2.CV_32FC1
    )
    # map2x, map2y: maps for the Right camera (R2/P2)
    map2x, map2y = cv2.initUndistortRectifyMap(
        mtxR, distR, R2, P2, img_shape, cv2.CV_32FC1
    )

    print("Successfully computed rectification maps.")
    return (map1x, map1y), (map2x, map2y)


def find_raw_sequences(raw_root):
    """
    Finds sequence folders and returns the raw image directories.
    Returns dict: {seq_name: (raw_left_dir, raw_right_dir)}
    """
    seq_map = {}
    
    seqs = sorted([d for d in os.listdir(raw_root) if d.startswith("seq_")])

    for s in seqs:
        seq_path = os.path.join(raw_root, s)
        
        # Adjust paths based on your data structure, assuming KITTI-like layout
        left_dir = os.path.join(seq_path, "image_02", "data") 
        right_dir = os.path.join(seq_path, "image_03", "data")
        
        if os.path.isdir(left_dir) and os.path.isdir(right_dir):
            seq_map[s] = (left_dir, right_dir)
        else:
            print(f"Warning: Skipping {s}, expected raw subdirectories not found.")
            
    return seq_map


def process_sequence_rectification(seq_name, raw_left_dir, raw_right_dir, mapsL, mapsR):
    """
    Applies rectification maps to all images in a sequence and saves them.
    """
    # 1. Define input/output paths
    left_files = sorted(glob.glob(os.path.join(raw_left_dir, "*.png")))
    right_files = sorted(glob.glob(os.path.join(raw_right_dir, "*.png")))
    
    if not left_files or not right_files:
        print(f"[{seq_name}] No images found to rectify.")
        return

    n = min(len(left_files), len(right_files))
    
    # Output structure matches the input structure to make Step 3 happy
    out_seq_root = os.path.join(OUT_RECTIFIED_ROOT, seq_name)
    out_left_dir = os.path.join(out_seq_root, "image_02", "data")
    out_right_dir = os.path.join(out_seq_root, "image_03", "data")
    
    os.makedirs(out_left_dir, exist_ok=True)
    os.makedirs(out_right_dir, exist_ok=True)
    
    map1x, map1y = mapsL
    map2x, map2y = mapsR
    
    print(f"[{seq_name}] Rectifying and saving {n} image pairs...")

    for i in range(n):
        lf = left_files[i]
        rf = right_files[i]
        
        # Read the raw images
        imgL = cv2.imread(lf, cv2.IMREAD_COLOR)
        imgR = cv2.imread(rf, cv2.IMREAD_COLOR)
        
        if imgL is None or imgR is None:
            continue
            
        # Apply rectification (Undistortion + Rotation/Shear)
        rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
        
        # Save to the new rectified structure
        base_name = os.path.basename(lf)
        cv2.imwrite(os.path.join(out_left_dir, base_name), rectL)
        cv2.imwrite(os.path.join(out_right_dir, base_name), rectR)

    print(f"[{seq_name}] Rectification complete. Saved to {out_seq_root}")


def main():
    print("\n=== STEP 2: BATCH RECTIFICATION OF RAW IMAGES ===")

    if not os.path.isfile(CALIB_NPZ):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_NPZ}. Run calibration script first.")

    # Find sequences in the RAW data folder
    seq_map = find_raw_sequences(RAW_ROOT)
    if not seq_map:
        raise RuntimeError(f"No raw sequences found in {RAW_ROOT}. Cannot proceed with rectification.")

    # Determine image shape from the first image
    first_left_dir = next(iter(seq_map.values()))[0]
    first_image = glob.glob(os.path.join(first_left_dir, "*.png"))
    if not first_image:
        raise RuntimeError("Could not find any image to determine image size.")
        
    test_img = cv2.imread(first_image[0], cv2.IMREAD_COLOR)
    if test_img is None:
        raise RuntimeError("Failed to read test image for size check.")
        
    img_shape = (test_img.shape[1], test_img.shape[0]) # (width, height)
    print(f"Detected image resolution: {img_shape[0]}x{img_shape[1]}")

    # Load calibration and compute maps only once
    mapsL, mapsR = load_calibration_maps(CALIB_NPZ, img_shape)
    
    # Ensure output root exists
    os.makedirs(OUT_RECTIFIED_ROOT, exist_ok=True)

    # Process all sequences
    for seq, (raw_ld, raw_rd) in seq_map.items():
        process_sequence_rectification(seq, raw_ld, raw_rd, mapsL, mapsR)

    print("\nSUCCESS: Batch Rectification is complete.")
    print("You can now run the Disparity/Depth script (Step 3) with full confidence.")


if __name__ == "__main__":
    main()