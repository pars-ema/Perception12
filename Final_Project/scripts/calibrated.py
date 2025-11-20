"""
Stereo Camera Calibration
Final Project - Perception of Autonomous Systems
DTU Course 34759
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path


# ======================== CONFIGURATION ========================

# Checkerboard patterns to detect (columns, rows of INNER corners)
CHECKERBOARD_PATTERNS = [
    (7, 5),     # Small pattern: 7 columns x 5 rows
    (7, 11),    # Medium pattern: 7 columns x 11 rows
    (15, 5),    # Large pattern: 15 columns x 5 rows
]

# Physical size of checkerboard squares in meters
SQUARE_SIZE_METERS = 0.025  # 25mm = 0.025m

# Input paths
LEFT_IMAGES_PATH = 'Final_Project/data/34759_final_project_raw/calib/image_02/data'
RIGHT_IMAGES_PATH = 'Final_Project/data/34759_final_project_raw/calib/image_03/data'

# Output path
OUTPUT_PATH = 'Final_Project/results/calibration'

# Reference calibration for comparison (optional)
REFERENCE_CALIBRATION = 'Final_Project/data/34759_final_project_raw/calib/stereo_calibration.npz'

# ======================== END CONFIGURATION ========================


def get_image_pairs(left_path, right_path):
    """
    Load and sort stereo image pairs from left and right directories
    
    Returns:
        list of tuples: [(left_img_path, right_img_path), ...]
    """
    left_images = sorted(glob.glob(os.path.join(left_path, '*.png')))
    right_images = sorted(glob.glob(os.path.join(right_path, '*.png')))
    
    if not left_images or not right_images:
        raise FileNotFoundError(f"No images found in {left_path} or {right_path}")
    
    if len(left_images) != len(right_images):
        print(f"Warning: Unequal number of images - Left: {len(left_images)}, Right: {len(right_images)}")
    
    return list(zip(left_images, right_images))


def create_object_points(pattern, square_size):
    """
    Create 3D world coordinates for checkerboard corners
    
    Args:
        pattern: Tuple of (columns, rows) 
        square_size: Physical size of squares in meters
    
    Returns:
        numpy array: 3D coordinates of corners
    """
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_corners(image_pair, pattern, criteria):
    """
    Detect and refine checkerboard corners in stereo image pair
    
    Args:
        image_pair: Tuple of (left_image_path, right_image_path)
        pattern: Tuple of (columns, rows)
        criteria: Corner refinement criteria
    
    Returns:
        Tuple: (success, left_corners, right_corners, gray_left, gray_right)
    """
    left_path, right_path = image_pair
    
    # Read images
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    
    if img_left is None or img_right is None:
        return False, None, None, None, None
    
    # Convert to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern, None)
    
    # Only proceed if found in both images
    if not (ret_left and ret_right):
        return False, None, None, gray_left, gray_right
    
    # Refine corner locations to sub-pixel accuracy
    corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
    
    return True, corners_left, corners_right, gray_left, gray_right


def collect_calibration_data(image_pairs, patterns, square_size):
    """
    Collect corner detections from all image pairs for all patterns
    
    Returns:
        Tuple: (object_points, image_points_left, image_points_right, image_shape, visualizations)
    """
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Storage for calibration data
    object_points = []
    image_points_left = []
    image_points_right = []
    
    # Storage for visualization
    visualization_data = []
    
    image_shape = None
    total_detections = 0
    
    print("\n" + "="*70)
    print("DETECTING CHECKERBOARD CORNERS")
    print("="*70 + "\n")
    
    for pattern in patterns:
        print(f"Pattern {pattern[0]}x{pattern[1]}:")
        
        # Create 3D object points for this pattern
        objp = create_object_points(pattern, square_size)
        
        pattern_count = 0
        
        for idx, pair in enumerate(image_pairs):
            success, corners_l, corners_r, gray_l, gray_r = detect_corners(pair, pattern, criteria)
            
            if image_shape is None and gray_l is not None:
                image_shape = gray_l.shape[::-1]
            
            if success:
                object_points.append(objp)
                image_points_left.append(corners_l)
                image_points_right.append(corners_r)
                pattern_count += 1
                total_detections += 1
                
                # Save visualization data
                if pattern_count == 1:  # Save first successful detection
                    img_l = cv2.imread(pair[0])
                    img_r = cv2.imread(pair[1])
                    img_l_vis = cv2.drawChessboardCorners(img_l, pattern, corners_l, True)
                    img_r_vis = cv2.drawChessboardCorners(img_r, pattern, corners_r, True)
                    visualization_data.append((img_l_vis, img_r_vis, pattern))
        
        print(f"  Found in {pattern_count} image pairs")
    
    print(f"\nTotal successful detections: {total_detections}")
    
    if total_detections < 10:
        print("WARNING: Less than 10 detections found. Calibration may be unreliable.")
    
    return object_points, image_points_left, image_points_right, image_shape, visualization_data


def perform_calibration(objpoints, imgpoints_left, imgpoints_right, image_shape):
    """
    Perform monocular and stereo camera calibration
    
    Returns:
        Dictionary containing all calibration parameters
    """
    print("\n" + "="*70)
    print("PERFORMING CAMERA CALIBRATION")
    print("="*70 + "\n")
    
    # Calibrate left camera
    print("Calibrating left camera...")
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_shape, None, None
    )
    print(f"  Reprojection error: {ret_left:.4f} pixels")
    
    # Calibrate right camera
    print("\nCalibrating right camera...")
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_shape, None, None
    )
    print(f"  Reprojection error: {ret_right:.4f} pixels")
    
    # Stereo calibration
    print("\nPerforming stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        image_shape, criteria=criteria, flags=flags
    )
    
    baseline = np.linalg.norm(T)
    
    print(f"  Stereo reprojection error: {ret_stereo:.4f} pixels")
    print(f"  Baseline: {baseline:.6f} m ({baseline*100:.2f} cm)")
    
    # Compute rectification parameters
    print("\nComputing rectification parameters...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right,
        image_shape, R, T, alpha=0
    )
    print("  Done")
    
    # Package results
    calibration = {
        'mtx_left': mtx_left,
        'dist_left': dist_left,
        'mtx_right': mtx_right,
        'dist_right': dist_right,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'baseline': baseline,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'image_shape': image_shape,
        'reprojection_error_left': ret_left,
        'reprojection_error_right': ret_right,
        'reprojection_error_stereo': ret_stereo
    }
    
    return calibration


def save_calibration(calibration, output_path):
    """
    Save calibration parameters to file
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING CALIBRATION RESULTS")
    print("="*70 + "\n")
    
    # Save as NumPy archive
    npz_file = os.path.join(output_path, 'stereo_calibration.npz')
    np.savez(npz_file, **calibration)
    print(f"Saved calibration to: {npz_file}")
    
    # Save as YAML for left camera
    yaml_left = os.path.join(output_path, 'camera_left.yaml')
    cv_file = cv2.FileStorage(yaml_left, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", calibration['mtx_left'])
    cv_file.write("distortion_coefficients", calibration['dist_left'])
    cv_file.release()
    print(f"Saved left camera to: {yaml_left}")
    
    # Save as YAML for right camera
    yaml_right = os.path.join(output_path, 'camera_right.yaml')
    cv_file = cv2.FileStorage(yaml_right, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", calibration['mtx_right'])
    cv_file.write("distortion_coefficients", calibration['dist_right'])
    cv_file.release()
    print(f"Saved right camera to: {yaml_right}")


def save_visualizations(visualization_data, output_path):
    """
    Save corner detection visualizations
    """
    if not visualization_data:
        return
    
    print("\nSaving visualizations...")
    
    for idx, (img_left, img_right, pattern) in enumerate(visualization_data):
        # Combine images side by side
        combined = np.hstack([img_left, img_right])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'LEFT CAMERA', (50, 50), font, 1.5, (0, 255, 0), 3)
        cv2.putText(combined, 'RIGHT CAMERA', (img_left.shape[1] + 50, 50), font, 1.5, (0, 255, 0), 3)
        cv2.putText(combined, f'Pattern: {pattern[0]}x{pattern[1]}', (50, combined.shape[0] - 30), 
                   font, 1.0, (255, 255, 255), 2)
        
        # Save
        output_file = os.path.join(output_path, f'corner_detection_{pattern[0]}x{pattern[1]}.png')
        cv2.imwrite(output_file, combined)
        print(f"  Saved: {output_file}")


def print_calibration_results(calibration):
    """
    Print calibration results in readable format
    """
    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70 + "\n")
    
    print("LEFT CAMERA:")
    print(f"  Focal length:    fx = {calibration['mtx_left'][0,0]:.2f} px")
    print(f"                   fy = {calibration['mtx_left'][1,1]:.2f} px")
    print(f"  Principal point: cx = {calibration['mtx_left'][0,2]:.2f} px")
    print(f"                   cy = {calibration['mtx_left'][1,2]:.2f} px")
    print(f"  Distortion:      {calibration['dist_left'].ravel()}")
    
    print("\nRIGHT CAMERA:")
    print(f"  Focal length:    fx = {calibration['mtx_right'][0,0]:.2f} px")
    print(f"                   fy = {calibration['mtx_right'][1,1]:.2f} px")
    print(f"  Principal point: cx = {calibration['mtx_right'][0,2]:.2f} px")
    print(f"                   cy = {calibration['mtx_right'][1,2]:.2f} px")
    print(f"  Distortion:      {calibration['dist_right'].ravel()}")
    
    print("\nSTEREO PARAMETERS:")
    print(f"  Baseline:        {calibration['baseline']:.6f} m ({calibration['baseline']*100:.2f} cm)")
    print(f"  Translation:     {calibration['T'].ravel()}")


def compare_with_reference(calibration, reference_path, output_path):
    """
    Compare calibration results with reference calibration
    """
    if not os.path.exists(reference_path):
        print(f"\nReference calibration not found: {reference_path}")
        return
    
    print("\n" + "="*70)
    print("COMPARISON WITH REFERENCE CALIBRATION")
    print("="*70 + "\n")
    
    ref = np.load(reference_path)
    
    # Extract reference values
    ref_fx_l = ref['mtx_left'][0, 0]
    ref_fy_l = ref['mtx_left'][1, 1]
    ref_cx_l = ref['mtx_left'][0, 2]
    ref_cy_l = ref['mtx_left'][1, 2]
    ref_baseline = ref.get('baseline', np.linalg.norm(ref['T']))
    
    # Extract computed values
    fx_l = calibration['mtx_left'][0, 0]
    fy_l = calibration['mtx_left'][1, 1]
    cx_l = calibration['mtx_left'][0, 2]
    cy_l = calibration['mtx_left'][1, 2]
    baseline = calibration['baseline']
    
    # Print comparison
    print(f"{'Parameter':<20} {'Your Value':>15} {'Reference':>15} {'Difference':>15}")
    print("-" * 65)
    print(f"{'fx (left)':<20} {fx_l:>15.2f} {ref_fx_l:>15.2f} {abs(fx_l - ref_fx_l):>15.2f}")
    print(f"{'fy (left)':<20} {fy_l:>15.2f} {ref_fy_l:>15.2f} {abs(fy_l - ref_fy_l):>15.2f}")
    print(f"{'cx (left)':<20} {cx_l:>15.2f} {ref_cx_l:>15.2f} {abs(cx_l - ref_cx_l):>15.2f}")
    print(f"{'cy (left)':<20} {cy_l:>15.2f} {ref_cy_l:>15.2f} {abs(cy_l - ref_cy_l):>15.2f}")
    print(f"{'Baseline (m)':<20} {baseline:>15.6f} {ref_baseline:>15.6f} {abs(baseline - ref_baseline):>15.6f}")
    
    # Save comparison to file
    comp_file = os.path.join(output_path, 'calibration_comparison.txt')
    with open(comp_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CALIBRATION COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write(f"Reprojection error: {calibration['reprojection_error_stereo']:.4f} pixels\n")
        f.write(f"Your baseline: {baseline:.6f} m\n")
        f.write(f"Reference baseline: {ref_baseline:.6f} m\n")
        f.write(f"Baseline difference: {abs(baseline - ref_baseline):.6f} m\n\n")
        f.write("Small differences are expected due to:\n")
        f.write("  - Different corner detection accuracy\n")
        f.write("  - Optimization convergence variations\n")
        f.write("  - Sub-pixel refinement differences\n")
    
    print(f"\nComparison saved to: {comp_file}")


def main():
    """
    Main calibration workflow
    """
    print("\n" + "="*70)
    print("STEREO CAMERA CALIBRATION")
    print("DTU Course 34759 - Perception for Autonomous Systems")
    print("="*70)
    
    # Load image pairs
    print("\nLoading image pairs...")
    image_pairs = get_image_pairs(LEFT_IMAGES_PATH, RIGHT_IMAGES_PATH)
    print(f"Found {len(image_pairs)} stereo image pairs")
    
    # Collect calibration data
    objpoints, imgpoints_left, imgpoints_right, image_shape, visualizations = collect_calibration_data(
        image_pairs, CHECKERBOARD_PATTERNS, SQUARE_SIZE_METERS
    )
    
    if len(objpoints) == 0:
        print("\nERROR: No checkerboard corners detected!")
        print("Please check:")
        print("  1. Image paths are correct")
        print("  2. Checkerboard pattern size is correct")
        print("  3. Images contain visible checkerboards")
        return
    
    # Perform calibration
    calibration = perform_calibration(objpoints, imgpoints_left, imgpoints_right, image_shape)
    
    # Save results
    save_calibration(calibration, OUTPUT_PATH)
    save_visualizations(visualizations, OUTPUT_PATH)
    
    # Print results
    print_calibration_results(calibration)
    
    # Compare with reference
    compare_with_reference(calibration, REFERENCE_CALIBRATION, OUTPUT_PATH)
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {OUTPUT_PATH}/")
    print("\nReady for next step: Image Rectification")
    print()


if __name__ == '__main__':
    main()