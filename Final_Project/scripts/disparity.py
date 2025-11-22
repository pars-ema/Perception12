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

# Valid disparity threshold for depth
DISP_VALID_THRESH = 0.5
# ======================================================


def find_rectified_sequences():
    """
    Find sequences in candidate roots.
    Supports both layouts:
      seq_xx/image_02_left/provided & image_03_right/provided
      seq_xx/image_02/data & image_03/data
    Returns dict: {seq_name: (left_dir, right_dir)}
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
    Load calibration:
      f from P1[0,0]
      B from |T_x|
    """
    data = np.load(calib_path, allow_pickle=True)
    P1 = data["P1"]
    T = data["T"].reshape(-1)

    f = float(P1[0, 0])
    B = float(abs(T[0])) if T.size >= 1 else 0.0
    return f, B


def compute_disparity_sgbm(imgL_gray, imgR_gray, stereo):
    """
    Compute disparity with SGBM + smoothing.
    Returns float32 disparity in pixels.
    """
    disp = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0

    # Cleanup / smoothing
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


def visualize_disparity(disp):
    """
    Disparity visualization for report.
    """
    d = disp.copy()
    d[d < 0] = 0
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX)
    d = d.astype(np.uint8)
    d = cv2.equalizeHist(d)
    return d


def visualize_depth_near_white(depth):
    """
    Near objects appear brighter.
    Uses inverse depth + fixed clipping.
    """
    d = depth.copy()
    d[np.isinf(d)] = DEPTH_MAX_M
    d = np.clip(d, DEPTH_MIN_M, DEPTH_MAX_M)

    inv = 1.0 / d
    vis = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.bilateralFilter(vis, 7, 50, 50)
    return vis


def process_sequence(seq_name, left_dir, right_dir, f, B, stereo):
    left_imgs  = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_imgs = sorted(glob.glob(os.path.join(right_dir, "*.png")))

    if not left_imgs or not right_imgs:
        print(f"Skipping {seq_name}: no rectified stereo images found.")
        return

    n = min(len(left_imgs), len(right_imgs))
    left_imgs, right_imgs = left_imgs[:n], right_imgs[:n]

    out_seq   = os.path.join(OUT_ROOT, seq_name)
    out_disp  = os.path.join(out_seq, "disparity")
    out_depth = os.path.join(out_seq, "depth")
    os.makedirs(out_disp, exist_ok=True)
    os.makedirs(out_depth, exist_ok=True)

    print(f"[{seq_name}] Processing {n} rectified pairs...")

    for lf, rf in zip(left_imgs, right_imgs):
        imgL = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            continue

        disp  = compute_disparity_sgbm(imgL, imgR, stereo)
        depth = disparity_to_depth(disp, f, B)

        base = os.path.splitext(os.path.basename(lf))[0]

        # Save raw arrays
        np.save(os.path.join(out_disp,  base + ".npy"), disp)
        np.save(os.path.join(out_depth, base + ".npy"), depth)

        # Save visuals
        cv2.imwrite(os.path.join(out_disp,  base + ".png"), visualize_disparity(disp))
        cv2.imwrite(os.path.join(out_depth, base + ".png"), visualize_depth_near_white(depth))

    print(f"[{seq_name}] saved disparity + depth to {out_seq}")


def main():
    print("\n=== STEP 3: DISPARITY + DEPTH COMPUTATION ===")

    if not os.path.isfile(CALIB_NPZ):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_NPZ}")

    seq_map = find_rectified_sequences()
    if not seq_map:
        raise RuntimeError("No rectified sequences found. Check rectified root paths.")

    f, B = load_calibration(CALIB_NPZ)
    if B <= 0:
        raise RuntimeError("Baseline B could not be read from calibration.")

    print(f"Loaded calibration: f={f:.2f}px, B={B:.4f}m")

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Create SGBM ONCE (faster + consistent)
    stereo = cv2.StereoSGBM_create(**SGBM_PARAMS)

    for seq, (ld, rd) in seq_map.items():
        process_sequence(seq, ld, rd, f, B, stereo)


if __name__ == "__main__":
    main()
