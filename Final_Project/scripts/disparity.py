import os
import glob
import numpy as np
import cv2

# ======================= CONFIG =======================
RECTIFIED_ROOT_CANDIDATES = [
    "Final_Project/data/34759_final_project_rect",   # your rectified data root
    "Final_Project/results/rectified",               # if you saved rectified here
    "Final_Project/data/34759_final_project_raw"     # fallback (should not be used ideally)
]

CALIB_NPZ = "Final_Project/models/stereo_calibration.npz"
OUT_ROOT = "Final_Project/results/step3_disparity_depth"

# StereoSGBM params (good default; you can mention them in report)
SGBM_PARAMS = dict(
    minDisparity=-16,            # allow slight negative disparities
    numDisparities=128,          # must be divisible by 16
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

# Depth clamp for visualization only (meters)
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 80.0

# ======================================================


def find_rectified_sequences():
    """
    Returns a dict:
      seq_name -> (left_folder, right_folder)
    It tries multiple common folder layouts.
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

            # layout A: seq_xx/image_02_left/provided + image_03_right/provided
            lA = os.path.join(seq_path, "image_02_left", "provided")
            rA = os.path.join(seq_path, "image_03_right", "provided")

            # layout B: seq_xx/image_02/data + image_03/data
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
    Loads Q, P1, T from your calibration.py output.
    """
    data = np.load(calib_path, allow_pickle=True)
    Q = data["Q"]
    P1 = data["P1"]
    T = data["T"]

    # focal length from rectified projection matrix
    f = float(P1[0, 0])

    # baseline from translation (meters)
    T = T.reshape(-1)
    B = float(abs(T[0])) if T.size >= 1 else 0.0

    return Q, f, B


def compute_disparity_sgbm(imgL_gray, imgR_gray):
    stereo = cv2.StereoSGBM_create(**SGBM_PARAMS)
    disp = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0

    # basic cleanup without contrib
    disp = cv2.medianBlur(disp, 5)
    disp = cv2.bilateralFilter(disp, d=7, sigmaColor=50, sigmaSpace=50)

    return disp


def disparity_to_depth(disp, f, B):
    """
    Z = f * B / d
    disp in pixels, f in pixels, B in meters => Z in meters
    """
    depth = np.full_like(disp, np.inf, dtype=np.float32)

    valid = disp > 0.5  # avoid tiny/negative disparities
    depth[valid] = (f * B) / disp[valid]

    return depth


def visualize_disparity(disp):
    disp_vis = disp.copy()
    disp_vis[disp_vis < 0] = 0
    disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
    return disp_vis.astype(np.uint8)


def visualize_depth_near_white(depth):
    """
    Near objects -> white.
    We visualize inverse depth and clamp to a range.
    """
    depth_clip = depth.copy()
    depth_clip[np.isinf(depth_clip)] = DEPTH_MAX_M
    depth_clip = np.clip(depth_clip, DEPTH_MIN_M, DEPTH_MAX_M)

    inv_depth = 1.0 / depth_clip  # near -> larger
    inv_vis = cv2.normalize(inv_depth, None, 0, 255, cv2.NORM_MINMAX)

    return inv_vis.astype(np.uint8)


def process_sequence(seq_name, left_dir, right_dir, f, B):
    left_imgs = sorted(glob.glob(os.path.join(left_dir, "*.png")))
    right_imgs = sorted(glob.glob(os.path.join(right_dir, "*.png")))

    if not left_imgs or not right_imgs:
        print(f"⚠️ Skipping {seq_name}: no rectified stereo images found.")
        return

    n = min(len(left_imgs), len(right_imgs))
    left_imgs, right_imgs = left_imgs[:n], right_imgs[:n]

    out_seq = os.path.join(OUT_ROOT, seq_name)
    out_disp = os.path.join(out_seq, "disparity")
    out_depth = os.path.join(out_seq, "depth")
    os.makedirs(out_disp, exist_ok=True)
    os.makedirs(out_depth, exist_ok=True)

    print(f"[{seq_name}] Processing {n} rectified pairs...")

    sample_indices = {0, n//2}  # for report screenshots

    for i, (lf, rf) in enumerate(zip(left_imgs, right_imgs)):
        imgL = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            continue

        disp = compute_disparity_sgbm(imgL, imgR)
        depth = disparity_to_depth(disp, f, B)

        # save raw
        base = os.path.splitext(os.path.basename(lf))[0]
        np.save(os.path.join(out_disp, base + ".npy"), disp)
        np.save(os.path.join(out_depth, base + ".npy"), depth)

        # save visualizations
        disp_vis = visualize_disparity(disp)
        depth_vis = visualize_depth_near_white(depth)

        cv2.imwrite(os.path.join(out_disp, base + ".png"), disp_vis)
        cv2.imwrite(os.path.join(out_depth, base + ".png"), depth_vis)

        if i in sample_indices:
            dmin, dmax = float(np.nanmin(disp)), float(np.nanmax(disp))
            zmin = float(np.nanmin(depth[np.isfinite(depth)])) if np.any(np.isfinite(depth)) else np.inf
            zmax = float(np.nanmax(depth[np.isfinite(depth)])) if np.any(np.isfinite(depth)) else np.inf
            print(f"  sample '{base}.png': disp range [{dmin:.2f}, {dmax:.2f}] px, depth range [{zmin:.2f}, {zmax:.2f}] m")

    print(f"[{seq_name}] ✅ saved disparity + depth to {out_seq}")


def main():
    print("\n=== STEP 3: DISPARITY + DEPTH COMPUTATION ===")

    if not os.path.isfile(CALIB_NPZ):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_NPZ}")

    seq_map = find_rectified_sequences()
    if not seq_map:
        raise RuntimeError("No rectified sequences found. Check your rectified root folder paths.")

    Q, f, B = load_calibration(CALIB_NPZ)
    if B <= 0:
        raise RuntimeError("Baseline B could not be read from calibration. Check T in npz file.")

    print(f"Loaded calibration: f={f:.2f}px, B={B:.4f}m")

    os.makedirs(OUT_ROOT, exist_ok=True)

    for seq, (ld, rd) in seq_map.items():
        process_sequence(seq, ld, rd, f, B)

    print("\nDONE! Step 3 fully completed for all sequences.\n")


if __name__ == "__main__":
    main()
