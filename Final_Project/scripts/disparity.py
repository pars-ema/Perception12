import os
import glob
import numpy as np
import cv2

# ======================= CONFIG =======================
RECTIFIED_ROOT_CANDIDATES = [
    "Final_Project/data/34759_final_project_rect",   
    "Final_Project/results/rectified",               
    "Final_Project/data/34759_final_project_raw"     
]

CALIB_NPZ = "Final_Project/models/stereo_calibration.npz"
OUT_ROOT = "Final_Project/results/disparity_depth"

SGBM_PARAMS = dict(
    minDisparity=-16,          
    numDisparities=128,        
    blockSize=5,
    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 80.0

# Valid disparity threshold for depth/point cloud
DISP_VALID_THRESH = 0.5
# ======================================================


def find_rectified_sequences():
    """
    Find sequences in candidate roots.
    Supports both layouts. Returns dict: {seq_name: (left_dir, right_dir)}
    """
    for root in RECTIFIED_ROOT_CANDIDATES:
        if not os.path.isdir(root):
            continue

        seqs = sorted([d for d in os.listdir(root) if d.startswith("seq_")])
        if not seqs:
            continue

        seq_map = {}
        for s in seqs:
            seq_path = os.path.join(root, s)

            lA = os.path.join(seq_path, "image_02_left", "provided")
            rA = os.path.join(seq_path, "image_03_right", "provided")

            lB = os.path.join(seq_path, "image_02", "data")
            rB = os.path.join(seq_path, "image_03", "data")

            if os.path.isdir(lA) and os.path.isdir(rA):
                seq_map[s] = (lA, rA)
            elif os.path.isdir(lB) and os.path.isdir(rB):
                seq_map[s] = (lB, rB)

        if seq_map:
            return seq_map

    return {}


def load_calibration(calib_path):
    """
    Load calibration parameters required for depth and 3D reconstruction:
      f: Focal length (P1[0,0])
      B: Baseline (abs(T_x))
      cx: Principal point x (P1[0,2])
      cy: Principal point y (P1[1,2])
    """
    data = np.load(calib_path, allow_pickle=True)
    P1 = data["P1"]
    T = data["T"].reshape(-1)

    f = float(P1[0, 0])
    B = float(abs(T[0])) if T.size >= 1 else 0.0
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    
    return f, B, cx, cy


def compute_disparity_sgbm(imgL_gray, imgR_gray, stereo):
    """
    Compute disparity with SGBM + smoothing.
    Returns float32 disparity in pixels.
    """
    # The result is 16-bit fixed-point: disparity * 16
    disp = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0

    # Cleanup / smoothing (This is why your results improved)
    disp = cv2.medianBlur(disp, 5)
    disp = cv2.bilateralFilter(disp, d=9, sigmaColor=60, sigmaSpace=60)

    return disp


def disparity_to_depth(disp, f, B):
    """
    Depth Z = f * B / disp
    """
    depth = np.full_like(disp, np.inf, dtype=np.float32)
    valid = disp > DISP_VALID_THRESH
    depth[valid] = (f * B) / disp[valid]
    return depth


def disparity_to_point_cloud(disp, f, B, cx, cy, imgL_color):
    """
    Projects the disparity map to 3D points (X, Y, Z) and extracts colors.
    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    Z = f * B / disp
    """
    h, w = disp.shape
    
    # 1. Calculate Z (Depth)
    Z = disparity_to_depth(disp, f, B)

    # 2. Create u, v coordinate grids
    # u (x-axis pixel coordinate), v (y-axis pixel coordinate)
    v, u = np.indices((h, w), dtype=np.float32)

    # 3. Calculate X and Y coordinates (3D projection formulas)
    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    
    # 4. Filter invalid points (where depth is inf or outside range)
    valid_mask = (Z != np.inf) & (Z > DEPTH_MIN_M) & (Z < DEPTH_MAX_M)

    # 5. Extract valid points and reshape
    points_3d = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=1)
    
    # 6. Extract colors (OpenCV uses BGR)
    colors = imgL_color[valid_mask]
    
    return points_3d, colors


def save_point_cloud(filename, points_3d, colors):
    """
    Saves the 3D point cloud data and color information to a standard PLY file.
    """
    
    # Create the structured array for PLY format
    # Using 'f4' for float32 for space (standard for PLY)
    points = np.zeros(points_3d.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    # Fill geometric data
    points['x'] = points_3d[:, 0]
    points['y'] = points_3d[:, 1]
    points['z'] = points_3d[:, 2]
    
    # Fill color data (OpenCV uses BGR, we need RGB for standard visualization)
    points['red'] = colors[:, 2]
    points['green'] = colors[:, 1]
    points['blue'] = colors[:, 0]

    # --- Writing the PLY file (ASCII format for easy inspection) ---
    header = [
        'ply',
        'format ascii 1.0',
        'element vertex %d' % points.shape[0],
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'end_header'
    ]
    
    with open(filename, 'w') as f:
        for line in header:
            f.write(line + '\n')
        
        # Write point data
        for i in range(points.shape[0]):
            p = points[i]
            # Format: X Y Z R G B
            f.write(f"{p['x']:.4f} {p['y']:.4f} {p['z']:.4f} {p['red']} {p['green']} {p['blue']}\n")
    
    print(f"3D Point Cloud saved to {filename} ({points.shape[0]} points).")
    # 


def visualize_disparity(disp):
    """
    Disparity visualization for report (normalized, equalized, grayscale).
    """
    d = disp.copy()
    # Mask out invalid (negative or near-zero) disparities
    d[d < DISP_VALID_THRESH] = 0 
    
    # Normalize to 0-255 range
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    d = d.astype(np.uint8)
    
    # Histogram equalization for better contrast
    d = cv2.equalizeHist(d)
    return d


def visualize_depth_near_white(depth):
    """
    Near objects appear brighter. Uses inverse depth + fixed clipping.
    """
    d = depth.copy()
    d[np.isinf(d)] = DEPTH_MAX_M
    d = np.clip(d, DEPTH_MIN_M, DEPTH_MAX_M)

    # Convert to inverse depth (closer = larger value)
    inv = 1.0 / d
    
    # Normalize and smooth
    vis = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.bilateralFilter(vis, 7, 50, 50)
    return vis


def process_sequence(seq_name, left_dir, right_dir, f, B, cx, cy, stereo):
    left_imgs  = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_imgs = sorted(glob.glob(os.path.join(right_dir, "*.png")))

    if not left_imgs or not right_imgs:
        print(f"Skipping {seq_name}: no rectified stereo images found.")
        return

    n = min(len(left_imgs), len(right_imgs))
    left_imgs, right_imgs = left_imgs[:n], right_imgs[:n]

    out_seq      = os.path.join(OUT_ROOT, seq_name)
    out_disp     = os.path.join(out_seq, "disparity")
    out_depth    = os.path.join(out_seq, "depth")
    out_pointcloud = os.path.join(out_seq, "pointcloud") # New folder
    os.makedirs(out_disp, exist_ok=True)
    os.makedirs(out_depth, exist_ok=True)
    os.makedirs(out_pointcloud, exist_ok=True) # Create point cloud folder

    print(f"[{seq_name}] Processing {n} rectified pairs...")

    for lf, rf in zip(left_imgs, right_imgs):
        # Load color image for point cloud color
        imgL_color = cv2.imread(lf, cv2.IMREAD_COLOR) 
        # Load grayscale image for disparity computation
        imgL_gray = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        
        if imgL_gray is None or imgR_gray is None or imgL_color is None:
            continue

        disp  = compute_disparity_sgbm(imgL_gray, imgR_gray, stereo)
        depth = disparity_to_depth(disp, f, B)

        points_3d, colors = disparity_to_point_cloud(disp, f, B, cx, cy, imgL_color)

        base = os.path.splitext(os.path.basename(lf))[0]

        # 1. Save raw arrays
        np.save(os.path.join(out_disp,  base + ".npy"), disp)
        np.save(os.path.join(out_depth, base + ".npy"), depth)

        # 2. Save visuals
        cv2.imwrite(os.path.join(out_disp,  base + ".png"), visualize_disparity(disp))
        cv2.imwrite(os.path.join(out_depth, base + ".png"), visualize_depth_near_white(depth))

        # 3. Save Point Cloud
        pc_path = os.path.join(out_pointcloud, base + ".ply")
        save_point_cloud(pc_path, points_3d, colors)

    print(f"[{seq_name}] saved disparity, depth, and point clouds to {out_seq}")


def main():
    print("\n=== STEP 3: DISPARITY, DEPTH, AND POINT CLOUD COMPUTATION ===")

    if not os.path.isfile(CALIB_NPZ):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_NPZ}")

    seq_map = find_rectified_sequences()
    if not seq_map:
        raise RuntimeError("No rectified sequences found. Check rectified root paths.")

    # Load calibration parameters: f, B, cx, cy
    f, B, cx, cy = load_calibration(CALIB_NPZ)
    if B <= 0:
        raise RuntimeError("Baseline B could not be read from calibration.")

    print(f"Loaded calibration: f={f:.2f}px, B={B:.4f}m, cx={cx:.2f}, cy={cy:.2f}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Create SGBM ONCE
    stereo = cv2.StereoSGBM_create(**SGBM_PARAMS)

    for seq, (ld, rd) in seq_map.items():
        process_sequence(seq, ld, rd, f, B, cx, cy, stereo)
        
    print("\nProcessing complete. Check the 'pointcloud' subdirectory for PLY files.")


if __name__ == "__main__":
    main()