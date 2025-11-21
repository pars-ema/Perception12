import os
import cv2
import numpy as np
import glob
from math import hypot

# CRITICAL PARAMETER FOR PRUNING:
MAX_REPROJ_ERROR_FOR_PRUNING = 1.0 # Discard any board with an error higher than 1.0 pixel.

# ======================= USER CONFIGURABLE PARAMETERS =======================
calibration_images_path = 'Final_Project/data/34759_final_project_raw/calib/image_02/data'
SAVE_CALIBRATION = "Final_Project/models/stereo_calibration.npz"
CHESSBOARD_SIZE = (7, 5)
SQUARE_SIZE = 0.025
CHESSBOARD_CANDIDATES = [
    CHESSBOARD_SIZE,
    (8, 6),
    (6, 5),
    (9, 6),
]
MAX_MATCH_DISTANCE_PX = 80
DIAG_DIR = os.path.join(os.path.dirname(SAVE_CALIBRATION), "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)
# ======================= END OF USER CONFIGURABLE PARAMETERS =================

def _refine_corners(gray, corners):
    # Increased max_iter for sub-pixel refinement
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4))


def preprocess_versions(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(gray)
    g_blur = cv2.GaussianBlur(g_clahe, (5, 5), 0)
    edges = cv2.Canny(g_clahe, 50, 150)
    edges = cv2.dilate(edges, None)
    ath = cv2.adaptiveThreshold(g_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return [g_clahe, g_blur, edges, ath]


def detect_multiple_boards(gray_img, candidates):
    boards = []
    mask_global = np.ones_like(gray_img, dtype=np.uint8) * 255
    preps = preprocess_versions(gray_img)

    for prep in preps:
        img = prep.copy()
        mask = mask_global.copy()
        for rot in [0, 90, 180, 270]:
            if rot == 0:
                attempt = img.copy()
            else:
                if rot == 90:
                    attempt = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    attempt = cv2.rotate(img, cv2.ROTATE_180)
                else:
                    attempt = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if rot == 0:
                attempt_masked = cv2.bitwise_and(attempt, attempt, mask=mask)
            else:
                if rot == 90:
                    mrot = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                elif rot == 180:
                    mrot = cv2.rotate(mask, cv2.ROTATE_180)
                else:
                    mrot = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                attempt_masked = cv2.bitwise_and(attempt, attempt, mask=mrot)

            for cand in candidates:
                while True:
                    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                             cv2.CALIB_CB_NORMALIZE_IMAGE |
                             cv2.CALIB_CB_FAST_CHECK)
                    ret, corners = cv2.findChessboardCorners(attempt_masked, cand, flags)
                    if not ret:
                        break

                    if rot != 0:
                        corners_xy = corners.reshape(-1, 2)
                        h, w = attempt_masked.shape[:2]
                        if rot == 90:
                            inv = np.array([[pt[1], w - 1 - pt[0]] for pt in corners_xy], dtype=np.float32)
                        elif rot == 180:
                            inv = np.array([[w - 1 - pt[0], h - 1 - pt[1]] for pt in corners_xy], dtype=np.float32)
                        else:  # 270
                            inv = np.array([[h - 1 - pt[1], pt[0]] for pt in corners_xy], dtype=np.float32)
                        corners = inv.reshape(-1, 1, 2).astype(np.float32)

                    corners_refined = _refine_corners(gray_img, corners)
                    if corners_refined is None or corners_refined.size == 0:
                        break

                    pts = corners_refined.reshape(-1, 2)
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))

                    duplicate = False
                    for b in boards:
                        dd = hypot(b['centroid'][0] - cx, b['centroid'][1] - cy)
                        if dd < 15:
                            duplicate = True
                            break
                    if not duplicate:
                        boards.append({
                            'corners': corners_refined,
                            'size': cand,
                            'centroid': (cx, cy)
                        })

                    x_min = max(0, int(pts[:, 0].min()) - 5)
                    x_max = min(attempt_masked.shape[1] - 1, int(pts[:, 0].max()) + 5)
                    y_min = max(0, int(pts[:, 1].min()) - 5)
                    y_max = min(attempt_masked.shape[0] - 1, int(pts[:, 1].max()) + 5)
                    try:
                        hull = cv2.convexHull(pts.astype(np.int32))
                        cv2.fillConvexPoly(mask, hull.astype(np.int32), 0)
                    except Exception:
                        mask[y_min:y_max, x_min:x_max] = 0

                    if rot == 0:
                        attempt_masked = cv2.bitwise_and(attempt_masked, attempt_masked, mask=mask)
                    else:
                        if rot == 90:
                            mout = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                        elif rot == 180:
                            mout = cv2.rotate(mask, cv2.ROTATE_180)
                        else:
                            mout = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        attempt_masked = cv2.bitwise_and(attempt_masked, attempt_masked, mask=mout)

    return boards


def match_boards(left_boards, right_boards, max_dist=MAX_MATCH_DISTANCE_PX):
    matches = []
    used_r = set()
    for l in left_boards:
        lx, ly = l['centroid']
        best = None
        bestd = float('inf')
        for idx, r in enumerate(right_boards):
            if idx in used_r:
                continue
            rx, ry = r['centroid']
            d = hypot(lx - rx, ly - ry)
            if d < bestd:
                bestd = d
                best = idx
        if best is not None and bestd <= max_dist:
            used_r.add(best)
            matches.append((l, right_boards[best]))
    return matches

def prune_outliers(objPointss, imgPointssL, imgPointssR, mtxL, distL, rvecsL, tvecsL, threshold=1.0):
    """Prunes image points that result in a reprojection error above the threshold."""
    new_objPointss = []
    new_imgPointssL = []
    new_imgPointssR = []
    rejected_count = 0

    # Calculate reprojection error for each set of points
    for i in range(len(objPointss)):
        imgpoints_reproj, _ = cv2.projectPoints(objPointss[i], rvecsL[i], tvecsL[i], mtxL, distL)
        
        # Calculate the error for this board
        error = cv2.norm(imgPointssL[i], imgpoints_reproj, cv2.NORM_L2) / len(imgPointssL[i])
        
        if error <= threshold:
            new_objPointss.append(objPointss[i])
            new_imgPointssL.append(imgPointssL[i])
            new_imgPointssR.append(imgPointssR[i])
        else:
            rejected_count += 1
            
    return new_objPointss, new_imgPointssL, new_imgPointssR, rejected_count


def main():
    print("Starting stereo camera calibration...")

    base_dir = os.path.dirname(os.path.dirname(calibration_images_path))
    left_path = os.path.join(base_dir, 'image_02', 'data', '*.png')
    right_path = os.path.join(base_dir, 'image_03', 'data', '*.png')

    left_images = sorted(glob.glob(left_path))
    right_images = sorted(glob.glob(right_path))

    if len(left_images) == 0 or len(right_images) == 0:
        print("Error: No calibration images found.")
        print(f"Left path: {left_path}")
        print(f"Right path: {right_path}")
        return

    print(f"Found {len(left_images)} image pairs")

    objPointss = []
    imgPointssL = []
    imgPointssR = []

    total_detections = 0
    total_matches = 0

    # Data collection loop (with diagnostics re-enabled)
    for idx, (lf, rf) in enumerate(zip(left_images, right_images)):
        left_color = cv2.imread(lf)
        right_color = cv2.imread(rf)
        if left_color is None or right_color is None:
            print(f"Warning: Could not read image pair {idx}. Skipping.")
            continue

        left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

        left_boards = detect_multiple_boards(left_gray.copy(), CHESSBOARD_CANDIDATES)
        right_boards = detect_multiple_boards(right_gray.copy(), CHESSBOARD_CANDIDATES)

        total_detections += len(left_boards) + len(right_boards)
        matches = match_boards(left_boards, right_boards, MAX_MATCH_DISTANCE_PX)
        total_matches += len(matches)

        for (l, r) in matches:
            if l['corners'] is None or r['corners'] is None or l.get('size') != r.get('size'):
                continue
            size = l['size']
            expected_count = size[0] * size[1]
            if l['corners'].shape[0] != expected_count or r['corners'].shape[0] != expected_count:
                continue

            objp = np.zeros((expected_count, 3), np.float32)
            objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
            objp *= SQUARE_SIZE

            objPointss.append(objp)
            imgPointssL.append(l['corners'])
            imgPointssR.append(r['corners'])
        
        # Diagnostics image (RE-ENABLED)
        diag = np.hstack((left_color.copy(), right_color.copy()))
        w = left_color.shape[1]
        for b in left_boards:
            try:
                cv2.drawChessboardCorners(diag[:, :w], b['size'], b['corners'], True)
            except Exception:
                pass
        for b in right_boards:
            try:
                cv2.drawChessboardCorners(diag[:, w:], b['size'], b['corners'], True)
            except Exception:
                pass

        diag_path = os.path.join(DIAG_DIR, f'pair_{idx:03d}_{os.path.basename(lf)}')
        if not diag_path.lower().endswith('.png'):
            diag_path += '.png'
        cv2.imwrite(diag_path, diag) # Write the diagnostics image

    print(f"Total detections: {total_detections}, matched pairs: {total_matches}")
    initial_correspondences = len(objPointss)
    print(f"Final correspondences (initial set): {initial_correspondences}")
    if initial_correspondences == 0:
        print(f"No matched boards found. Check diagnostics in {DIAG_DIR}")
        return

    # Trim lists to equal length
    img_shape = (left_gray.shape[1], left_gray.shape[0])

    # --- Step 1: Preliminary Single Calibration for Pruning ---
    print("\n--- Calibration Pass 1: For Data Pruning ---")
    # Use standard 5-coefficient model (K1, K2, P1, P2, K3) for stability
    single_calib_flags = (cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_PRINCIPAL_POINT)
    retL_init, mtxL_init, distL_init, rvecsL_init, tvecsL_init = cv2.calibrateCamera(
        objPointss, imgPointssL, img_shape, None, None, flags=single_calib_flags
    )
    print(f"Initial Left Camera Reprojection Error: {retL_init:.3f}")

    # --- Step 2: Prune Outliers ---
    objPointss_pruned, imgPointssL_pruned, imgPointssR_pruned, rejected_count = prune_outliers(
        objPointss, imgPointssL, imgPointssR, 
        mtxL_init, distL_init, rvecsL_init, tvecsL_init, 
        threshold=MAX_REPROJ_ERROR_FOR_PRUNING
    )
    
    print(f"Pruning complete: Rejected {rejected_count} board(s) with error > {MAX_REPROJ_ERROR_FOR_PRUNING} pixel.")
    pruned_correspondences = len(objPointss_pruned)
    print(f"Final correspondences (pruned set): {pruned_correspondences}")

    if pruned_correspondences < 10:
        print("\n⚠️ WARNING: Not enough high-quality data remains for robust calibration. You must capture more data.")
    
    # Use the pruned data set for the final calibration pass
    objPointss = objPointss_pruned
    imgPointssL = imgPointssL_pruned
    imgPointssR = imgPointssR_pruned

    # --- Step 3: Final Single and Stereo Calibration with Pruned Data ---
    print("\n--- Calibration Pass 2: Final Calibration ---")
    
    print("Calibrating left camera...")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        objPointss, imgPointssL, img_shape, None, None, flags=single_calib_flags
    )
    print(f"Left camera reprojection error: {retL:.3f}")

    print("Calibrating right camera...")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        objPointss, imgPointssR, img_shape, None, None, flags=single_calib_flags
    )
    print(f"Right camera reprojection error: {retR:.3f}")

    print("Running stereoCalibrate...")
    # REVERTED: Using CALIB_USE_INTRINSIC_GUESS to allow the solver more flexibility with sparse data
    stereo_flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_TANGENT_DIST |
             cv2.CALIB_FIX_PRINCIPAL_POINT)
             
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-7)
    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objPointss, imgPointssL, imgPointssR,
        mtxL, distL, mtxR, distR, img_shape,
        criteria=criteria, flags=stereo_flags
    )
    
    print(f"Stereo reprojection error: {ret:.6f}")

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
    )

    # ----------------- SAVE calib.npz -----------------
    np.savez(SAVE_CALIBRATION,
             mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR,
             R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    print("Calibration saved to:", SAVE_CALIBRATION)

    # ----------------- SAVE calib.yaml -----------------
    yaml_file = os.path.splitext(SAVE_CALIBRATION)[0] + '.yaml'
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)
    fs.write("mtxL", mtxL)
    fs.write("distL", distL)
    fs.write("mtxR", mtxR)
    fs.write("distR", distR)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)
    fs.release()
    print("Calibration also saved to YAML:", yaml_file)

    # ----------------- SAVE final visualization files -----------------
    last_left = cv2.imread(left_images[0])
    last_right = cv2.imread(right_images[0])
    
    # Draw corners on a fresh copy
    corners_left = last_left.copy()
    corners_right = last_right.copy()

    for b in detect_multiple_boards(cv2.cvtColor(corners_left, cv2.COLOR_BGR2GRAY), CHESSBOARD_CANDIDATES):
        try:
            cv2.drawChessboardCorners(corners_left, b['size'], b['corners'], True)
        except Exception:
            pass
    for b in detect_multiple_boards(cv2.cvtColor(corners_right, cv2.COLOR_BGR2GRAY), CHESSBOARD_CANDIDATES):
        try:
            cv2.drawChessboardCorners(corners_right, b['size'], b['corners'], True)
        except Exception:
            pass
            
    cv2.imwrite(os.path.join(os.path.dirname(SAVE_CALIBRATION), 'calib_corners.png'),
                np.hstack((corners_left, corners_right)))

    # Save rectified image
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_shape, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_shape, cv2.CV_32FC1)
    rectL = cv2.remap(last_left, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(last_right, map2x, map2y, cv2.INTER_LINEAR)

    # Draw horizontal lines
    h = rectL.shape[0]
    step = max(20, h // 10)
    for y in range(0, h, step):
        cv2.line(rectL, (0, y), (rectL.shape[1], y), (0, 255, 0), 1)
        cv2.line(rectR, (0, y), (rectR.shape[1], y), (0, 255, 0), 1)

    cv2.imwrite(os.path.join(os.path.dirname(SAVE_CALIBRATION), 'calib_rectified.png'),
                np.hstack((rectL, rectR)))

    print(f"Per-pair diagnostics saved in: {DIAG_DIR}")
    print("Finished successfully. Please run this code and report the resulting Stereo reprojection error.")


if __name__ == "__main__":
    main()