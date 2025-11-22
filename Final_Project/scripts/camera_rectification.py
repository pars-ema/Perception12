import numpy as np
import cv2
import os 
import glob
from math import hypot

# ======================= CONFIG =======================
# Path to the calibration file
calibFile = "Final_Project/models/stereo_calibration.npz"

# Single image pair to use for visual confirmation of rectification quality
LEFT_IMAGE = "Final_Project/data/34759_final_project_raw/calib/image_02/data/000000.png"
RIGHT_IMAGE = "Final_Project/data/34759_final_project_raw/calib/image_03/data/000000.png"

# Output folder for the diagnostic image
OUTPUT_DIR = "Final_Project/models"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rectification_comparison_corrected.png")

# Chessboard size used during calibration - MUST match the size used in stereo_calibration.py
CHESSBOARD_SIZE = (7, 5) 

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ======================================================


def _load_image_with_fallback(path):
    """Safely loads an image, using the first available image in the directory as a fallback."""
    img = cv2.imread(path)
    if img is None:
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            print(f"Error: Image path '{path}' not found and directory '{directory}' does not exist.")
            return None
        candidates = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        if not candidates:
            print(f"Error: Couldn't read '{path}' and no image files found in directory '{directory}'.")
            return None
        fallback = os.path.join(directory, candidates[0])
        print(f"Warning: couldn't read '{path}'. Using fallback '{fallback}'.")
        img = cv2.imread(fallback)
        if img is None:
            print(f"Error: Failed to read fallback image '{fallback}'.")
            return None
    return img

def main():
    print("Loading calibration and images...")
    
    # --- Loading Calibration ---
    if not os.path.exists(calibFile):
        print(f"Error: Calibration file not found at {calibFile}")
        return

    data = np.load(calibFile)
    mtxL, distL = data['mtxL'], data['distL']
    mtxR, distR = data['mtxR'], data['distR']
    R1, P1 = data['R1'], data['P1']
    R2, P2 = data['R2'], data['P2']

    # --- Loading Stereo Images ---
    leftImg = _load_image_with_fallback(LEFT_IMAGE)
    rightImg = _load_image_with_fallback(RIGHT_IMAGE)
    
    if leftImg is None or rightImg is None:
        return

    h, w = leftImg.shape[:2]
    img_shape = (w, h)

    # --- Generating Rectification Maps ---
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_shape, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_shape, cv2.CV_32FC1)

    # --- Rectify Images ---
    rectL = cv2.remap(leftImg, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(rightImg, map2x, map2y, cv2.INTER_LINEAR)

    # --- Drawing Chess Board Corner Optimization (CORRECTED) ---
    grayL = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

    # 1. Detect corners on the RAW images
    print(f"Attempting to find chessboard {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}...")
    retL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE)

    if retL and retR:
        print("Corners found. Projecting points onto rectified images.")
        # Refine corners for sub-pixel accuracy
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4))
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4))
        
        # 2. Project raw corners to the RECTIFIED image plane using the calibration matrices
        
        # Left Image (uses R1 and P1)
        # UndistortPoints maps the raw points through distortion, then applies R1/P1 for rectification
        pts_rectL = cv2.undistortPoints(cornersL, mtxL, distL, R=R1, P=P1)
        for p in pts_rectL.reshape(-1, 2):
            x, y = int(p[0]), int(p[1])
            # Draw Red circle (BGR: Blue=0, Green=0, Red=255)
            cv2.circle(rectL, (x, y), 5, (0, 0, 255), -1) 

        # Right Image (uses R2 and P2)
        pts_rectR = cv2.undistortPoints(cornersR, mtxR, distR, R=R2, P=P2)
        for p in pts_rectR.reshape(-1, 2):
            x, y = int(p[0]), int(p[1])
            # Draw Red circle
            cv2.circle(rectR, (x, y), 5, (0, 0, 255), -1) 

        # 3. Calculate Vertical (Epipolar) Alignment Error
        y_left = pts_rectL.reshape(-1, 2)[:, 1]
        y_right = pts_rectR.reshape(-1, 2)[:, 1]
        
        # Calculate the absolute difference in y-coordinates for all corresponding points
        y_diff = np.abs(y_left - y_right)
        max_y_error = np.max(y_diff)
        avg_y_error = np.mean(y_diff)
        
        print(f"Epipolar Error (Vertical Alignment Check):")
        print(f"  Maximum vertical misalignment: {max_y_error:.4f} pixels")
        print(f"  Average vertical misalignment: {avg_y_error:.4f} pixels")
        
        if max_y_error > 1.0:
            print("WARNING: Maximum vertical error is high (> 1.0 pixel). Rectification may be suboptimal.")


    elif not retL or not retR:
        print("Warning: Could not find chessboard corners in one or both images. Cannot draw rectified points or calculate epipolar error.")


    # --- Draw Epipolar Lines ---
    step = 50
    for y in range (0, h, step):
        # Draw Green lines
        cv2.line(rectL, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(rectR, (0, y), (w, y), (0, 255, 0), 1)

    # --- Save Comparison ---
    comparison = np.hstack((rectL, rectR))
    cv2.imwrite(OUTPUT_FILE, comparison)

    print("\nSUCCESS: Rectification comparison saved to:", OUTPUT_FILE)
    print("Check this image. The Red dots (rectified corner points) should lie perfectly on the Green lines.")
    print("If the dots are on the lines and the vertical error is low, the calibration is good.")

if __name__ == "__main__":
    main()