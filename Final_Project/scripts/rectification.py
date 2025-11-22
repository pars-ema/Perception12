import os
import glob
import numpy as np
import cv2

# ======================= CONFIG =======================
CALIB_FILE = "Final_Project/models/stereo_calibration.npz"

# RAW calibration checkerboard images
CALIB_LEFT_DIR  = "Final_Project/data/34759_final_project_raw/calib/image_02/data"
CALIB_RIGHT_DIR = "Final_Project/data/34759_final_project_raw/calib/image_03/data"

# TEACHER rectified calibration images (if you want comparison)
# In your tree, teacher rectified data is under 34759_final_project_rect
TEACHER_LEFT_DIR  = "Final_Project/data/34759_final_project_rect/calib/image_02/data"
TEACHER_RIGHT_DIR = "Final_Project/data/34759_final_project_rect/calib/image_03/data"

OUT_DIR = "Final_Project/results/rectified"
OUT_YOURS = os.path.join(OUT_DIR, "our_rectified.png")
OUT_COMPARE = os.path.join(OUT_DIR, "teacher_vs_yours.png")
os.makedirs(OUT_DIR, exist_ok=True)

EPILINE_STEP = 50
# ======================================================

def load_calibration(npz_path):
    data = np.load(npz_path)
    return (data["mtxL"], data["distL"],
            data["mtxR"], data["distR"],
            data["R1"], data["R2"],
            data["P1"], data["P2"])

def list_pngs(d):
    return sorted(glob.glob(os.path.join(d, "*.png")))

def pick_first_pair(left_dir, right_dir):
    left_files = list_pngs(left_dir)
    right_files = list_pngs(right_dir)
    if not left_files or not right_files:
        raise FileNotFoundError("No calibration images found.")
    # Pair by sorted order (calib folders are aligned)
    return left_files[0], right_files[0]

def build_maps(calib, img_shape):
    mtxL, distL, mtxR, distR, R1, R2, P1, P2 = calib
    map1x, map1y = cv2.initUndistortRectifyMap(
        mtxL, distL, R1, P1, img_shape, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        mtxR, distR, R2, P2, img_shape, cv2.CV_32FC1
    )
    return map1x, map1y, map2x, map2y

def rectify_pair(imgL, imgR, maps):
    map1x, map1y, map2x, map2y = maps
    rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    return rectL, rectR

def draw_epipolar_lines(img, step=50):
    out = img.copy()
    h, w = out.shape[:2]
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
    return out

def main():
    calib = load_calibration(CALIB_FILE)

    # --- YOUR rectification on calib pair ---
    l_path, r_path = pick_first_pair(CALIB_LEFT_DIR, CALIB_RIGHT_DIR)
    imgL = cv2.imread(l_path)
    imgR = cv2.imread(r_path)

    h, w = imgL.shape[:2]
    maps = build_maps(calib, (w, h))
    rectL, rectR = rectify_pair(imgL, imgR, maps)

    rectL_lines = draw_epipolar_lines(rectL, EPILINE_STEP)
    rectR_lines = draw_epipolar_lines(rectR, EPILINE_STEP)

    yours = np.hstack([rectL_lines, rectR_lines])
    cv2.imwrite(OUT_YOURS, yours)
    print(" Saved YOUR rectified calib pair:", OUT_YOURS)

    # --- TEACHER vs YOURS comparison (optional but PDF asks for it) ---
    teach_left = list_pngs(TEACHER_LEFT_DIR)
    teach_right = list_pngs(TEACHER_RIGHT_DIR)
    if teach_left and teach_right:
        tL = cv2.imread(teach_left[0])
        tR = cv2.imread(teach_right[0])

        tL_lines = draw_epipolar_lines(tL, EPILINE_STEP)
        tR_lines = draw_epipolar_lines(tR, EPILINE_STEP)
        teacher = np.hstack([tL_lines, tR_lines])

        compare = np.vstack([yours, teacher])
        cv2.imwrite(OUT_COMPARE, compare)
        print(" Saved TEACHER vs YOURS comparison:", OUT_COMPARE)
    else:
        print("Teacher rectified calib images not found. Comparison skipped.")

if __name__ == "__main__":
    main()
