import os
import cv2
import numpy as np
import glob


# ======================= USER CONFIGURABLE PARAMETERS =======================


# Path to the directory containing calibration images
calibration_images_path = 'Final_Project/data/34759_final_project_raw/calib/image_02/data'
# Save calibration results to a file
SAVE_CALIBRATION = "Final_Project/models/stereo_calibration.npz"


# ======================= END OF USER CONFIGURABLE PARAMETERS =======================


CHESSBOARD_SIZE = (7, 5)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 0.025  # Size of a square in your defined unit 



def main():
    print("Starting stereo camera calibration...")

    # =========================================
    # 1. Preparing object points (3D points)
    # =========================================


    objPoints =np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objPoints[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 
                                0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    
    objPoints *= SQUARE_SIZE

    objPointss = []  # 3D points in real world space
    imgPointssL = []  # 2D points in image plane for left camera
    imgPointssR = []  # 2D points in image plane for right camera


    # =========================================
    # 2. Loading all image pairs
    # =========================================

    # If user passed a directory (e.g. '.../image_02/data'), handle that specially
    left_images = []
    right_images = []
    if os.path.isdir(calibration_images_path):
        # KITTI-style: calibration_images_path may be '.../image_02/data'
        if os.path.basename(calibration_images_path) == 'data' and os.path.basename(os.path.dirname(calibration_images_path)).startswith('image_'):
            base_dir = os.path.dirname(os.path.dirname(calibration_images_path))
            left_images = sorted(glob.glob(os.path.join(base_dir, 'image_02', 'data', '*.png')))
            right_images = sorted(glob.glob(os.path.join(base_dir, 'image_03', 'data', '*.png')))
        else:
            # treat provided dir as containing left images, try to find sibling folder for right
            left_images = sorted(glob.glob(os.path.join(calibration_images_path, '*.png')))
            sibling = os.path.join(os.path.dirname(calibration_images_path), 'image_03', 'data')
            right_images = sorted(glob.glob(os.path.join(sibling, '*.png')))
    else:
        # Try the user-provided pattern first (supports jpg/png replacement)
        left_images = sorted(glob.glob(calibration_images_path.replace('*.jpg', 'left_*.jpg').replace('*.png', 'left_*.png')))
        right_images = sorted(glob.glob(calibration_images_path.replace('*.jpg', 'right_*.jpg').replace('*.png', 'right_*.png')))

        # Fallback: detect KITTI-style folders (image_02 = left, image_03 = right)
        if len(left_images) == 0 or len(right_images) == 0:
            base_dir = calibration_images_path
            if base_dir.endswith('*.jpg') or base_dir.endswith('*.png'):
                base_dir = os.path.dirname(base_dir)  # strip the wildcard

            # Climb up until we find a folder named 'calib' (max 4 levels)
            for _ in range(4):
                if os.path.basename(base_dir) == 'calib':
                    break
                base_dir = os.path.dirname(base_dir)

            left_images = sorted(glob.glob(os.path.join(base_dir, 'image_02', 'data', '*.png')))
            right_images = sorted(glob.glob(os.path.join(base_dir, 'image_03', 'data', '*.png')))

    if len(left_images) == 0 or len(right_images) == 0:
        print("Error: No calibration images found! Please set `calibration_images_path` to point to your left/right images.")
        print("Tried patterns:")
        print(" - ", calibration_images_path.replace('*.jpg', 'left_*.jpg').replace('*.png', 'left_*.png'))
        print(" - ", os.path.join(os.path.dirname(os.path.dirname(calibration_images_path)), 'image_02', 'data', '*.png'))
        return

    if len(left_images) != len(right_images):
        print(f"Warning: number of left images ({len(left_images)}) != number of right images ({len(right_images)}). Using min count for pairing.")
    

    print(f"Found {len(left_images)} image pairs")

    # Save a sample stereo pair (raw) so user gets image output even if corners fail
    try:
        out_dir = os.path.dirname(SAVE_CALIBRATION) or '.'
        os.makedirs(out_dir, exist_ok=True)
        sampleL = cv2.imread(left_images[0])
        sampleR = cv2.imread(right_images[0])
        sample_comb = np.hstack((sampleL, sampleR))
        sample_out = os.path.join(out_dir, 'calib_sample_pair.png')
        cv2.imwrite(sample_out, sample_comb)
        print('Saved sample stereo pair to:', sample_out)
    except Exception as e:
        print('Could not save sample stereo pair:', e)



    # =========================================
    # 3. Detecting chessboard corners
    # =========================================

    for left_img_file, right_img_file in zip(left_images, right_images):

        imgL= cv2.imread(left_img_file, cv2.IMREAD_GRAYSCALE)
        imgR=cv2.imread(right_img_file, cv2.IMREAD_GRAYSCALE)

        retL, cornersL = cv2.findChessboardCorners(imgL, CHESSBOARD_SIZE, None)
        retR, cornersR = cv2.findChessboardCorners(imgR, CHESSBOARD_SIZE, None)

        if retL and retR:
           objPointss.append(objPoints)

           
           
           cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
           
           
           
           cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
           
           
           imgPointssL.append(cornersL)
           imgPointssR.append(cornersR)

    if len(objPointss) == 0:
        print("Error: No corners were detected in any image pair.")
        return
    
    print(f"Detected corners in {len(objPointss)} valid image pairs.")


    #Image Size 
    img_shape = imgL.shape[::-1]




    # ========================================
    # 4. Monocular calibration (left + right)
    # ========================================

    print("Calibrating left camera...")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        objPointss, imgPointssL, img_shape, None, None)
    
    print("Calibrating right camera...")
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        objPointss, imgPointssR, img_shape, None, None)
    


    # ========================================
    # 5. Stereo calibration 
    # ========================================

    print("Running stereoCalibrate... This may take a few seconds.")

    flags = cv2.CALIB_FIX_INTRINSIC  # Keep intrinsics fixed (recommended)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objPointss, imgPointssL, imgPointssR,
        mtxL, distL, mtxR, distR,
        img_shape,
        criteria=criteria,
        flags=flags
    )

    print(f"Stereo calibration reprojection error = {ret}")


    # ========================================
    # 6. Saving rectification 
    # ========================================

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR,
        img_shape, R, T, alpha=0
    )


    # ========================================
    # 7. Saving
    # ========================================
        
    np.savez(
    SAVE_CALIBRATION,
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R=R, T=T,
    R1=R1, R2=R2,
    P1=P1, P2=P2,
    Q=Q
    )

    print("Calibration complete!")
    print("Saved calibration to:", SAVE_CALIBRATION)


    # ==========================
    # 8. Save visualization images
    # ==========================
    try:
        out_dir = os.path.dirname(SAVE_CALIBRATION) or '.'
        os.makedirs(out_dir, exist_ok=True)

        # Use the last successful pair to draw corners and rectified result
        if len(imgPointssL) > 0 and len(left_images) > 0 and len(right_images) > 0:
            left_vis = cv2.imread(left_images[-1])
            right_vis = cv2.imread(right_images[-1])

            # Draw detected corners (if available)
            try:
                cv2.drawChessboardCorners(left_vis, CHESSBOARD_SIZE, imgPointssL[-1], True)
                cv2.drawChessboardCorners(right_vis, CHESSBOARD_SIZE, imgPointssR[-1], True)
            except Exception:
                pass

            corners_combined = np.hstack((left_vis, right_vis))
            corner_out = os.path.join(out_dir, 'calib_corners.png')
            cv2.imwrite(corner_out, corners_combined)
            print('Saved chessboard corners image to:', corner_out)

            # Create rectified images using stereo rectification maps
            try:
                map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_shape, cv2.CV_32FC1)
                map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_shape, cv2.CV_32FC1)

                rectL = cv2.remap(left_vis, map1x, map1y, interpolation=cv2.INTER_LINEAR)
                rectR = cv2.remap(right_vis, map2x, map2y, interpolation=cv2.INTER_LINEAR)

                # Draw horizontal lines to visualize epipolar alignment
                h = rectL.shape[0]
                step = max(20, h // 10)
                for y in range(0, h, step):
                    cv2.line(rectL, (0, y), (rectL.shape[1], y), (0, 255, 0), 1)
                    cv2.line(rectR, (0, y), (rectR.shape[1], y), (0, 255, 0), 1)

                rect_combined = np.hstack((rectL, rectR))
                rect_out = os.path.join(out_dir, 'calib_rectified.png')
                cv2.imwrite(rect_out, rect_combined)
                print('Saved rectified stereo image to:', rect_out)
            except Exception as e:
                print('Could not create rectified visualization:', e)
        else:
            print('No detected corners or image pairs available for visualization.')
    except Exception as e:
        print('Could not save visualization images:', e)


if __name__ == "__main__":
    main()







