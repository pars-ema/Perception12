import os
import glob
import json
import numpy as np
import cv2

# ======================= CONFIG =======================
CALIB_NPZ = "Final_Project/models/stereo_calibration.npz"

DEPTH_ROOT      = "Final_Project/results/disparity_depth"
DETECTION_ROOT  = "Final_Project/results/detection_2d"
OUT_ROOT        = "Final_Project/results/tracking_3d"

# Tracking hyperparams
MAX_ASSOCIATION_DIST_M = 3.0   
MAX_AGE  = 8                   
MIN_HITS = 3                   

# Depth filtering
DEPTH_MIN_M = 0.5
DEPTH_MAX_M = 80.0

# NMS to kill duplicate YOLO boxes per frame
NMS_IOU_THRESH = 0.5

# Visualization
SAVE_VIS = True
VIS_ROOT_CANDIDATES = [
    "Final_Project/data/34759_final_project_rect",   # prefer this first
    "Final_Project/results/rectified",
    "Final_Project/data/34759_final_project_raw",
]
# ======================================================


# ----------------------- Utilities -----------------------
def find_vis_left_dir(seq_name):
    """Find rectified left frames for visualization."""
    for root in VIS_ROOT_CANDIDATES:
        seq_path = os.path.join(root, seq_name)
        lA = os.path.join(seq_path, "image_02_left", "provided")
        lB = os.path.join(seq_path, "image_02", "data")
        if os.path.isdir(lA): return lA
        if os.path.isdir(lB): return lB
    return None


def load_calibration(calib_path):
    """Load f, cx, cy from left P1 matrix."""
    data = np.load(calib_path, allow_pickle=True)
    P1 = data["P1"]
    f  = float(P1[0, 0])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    return f, cx, cy


def load_detections(det_file):
    """Each det file is numpy object array of dicts."""
    arr = np.load(det_file, allow_pickle=True)
    return list(arr)


def bbox_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)
    return u, v


def depth_at_bbox_center(depth, box_xyxy, window=5):
    """Robust depth: median depth around bbox center."""
    u, v = bbox_center(box_xyxy)
    h, w = depth.shape
    r = window // 2
    u1, u2 = max(0, u - r), min(w, u + r + 1)
    v1, v2 = max(0, v - r), min(h, v + r + 1)

    patch = depth[v1:v2, u1:u2]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return np.inf
    return float(np.median(patch))


def detection_to_3d(det, depth, f, cx, cy):
    """Convert a 2D detection to 3D centroid using Z from depth map."""
    Z = depth_at_bbox_center(depth, det["box_xyxy"])
    if (not np.isfinite(Z)) or Z < DEPTH_MIN_M or Z > DEPTH_MAX_M:
        return None

    u, v = bbox_center(det["box_xyxy"])
    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    return np.array([X, Y, Z], dtype=np.float32)


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def iou_xyxy(a, b):
    """2D IoU between two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def nms_detections(dets, iou_thresh=0.5):
    """
    Simple class-aware NMS on YOLO detections.
    Keeps highest confidence boxes per class.
    """
    if len(dets) == 0:
        return dets

    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    keep = []

    for d in dets:
        ok = True
        for k in keep:
            if d["class_id"] == k["class_id"]:
                if iou_xyxy(d["box_xyxy"], k["box_xyxy"]) > iou_thresh:
                    ok = False
                    break
        if ok:
            keep.append(d)
    return keep


# ----------------------- Kalman Filter -----------------------
class Kalman3D:
    """
    Constant velocity KF:
      state=[X,Y,Z,Vx,Vy,Vz], meas=[X,Y,Z]
    """
    def __init__(self, x0):
        self.dt = 1.0
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.x[0:3, 0] = x0.reshape(3)

        self.F = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 3] = self.dt

        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        self.P = np.eye(6, dtype=np.float32) * 5.0
        self.Q = np.eye(6, dtype=np.float32) * 0.02
        self.R = np.eye(3, dtype=np.float32) * 0.6

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:3, 0].copy()

    def update(self, z):
        z = z.reshape(3, 1).astype(np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


# ----------------------- Track object -----------------------
class Track3D:
    _next_id = 0
    def __init__(self, det, pos3d):
        self.id = Track3D._next_id
        Track3D._next_id += 1

        self.kf = Kalman3D(pos3d)
        self.pos3d = pos3d

        self.class_id   = det["class_id"]
        self.class_name = det["class_name"]
        self.last_bbox  = det["box_xyxy"]
        self.last_conf  = det["confidence"]

        self.hits = 1
        self.time_since_update = 0
        self.history = [pos3d.tolist()]

    def predict(self):
        self.pos3d = self.kf.predict()
        self.time_since_update += 1

    def update(self, det, pos3d):
        self.kf.update(pos3d)
        self.pos3d = self.kf.x[0:3, 0].copy()

        self.class_id   = det["class_id"]
        self.class_name = det["class_name"]
        self.last_bbox  = det["box_xyxy"]
        self.last_conf  = det["confidence"]

        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.pos3d.tolist())

    def is_confirmed(self):
        return self.hits >= MIN_HITS

    def is_dead(self):
        return self.time_since_update > MAX_AGE


# ----------------------- Association -----------------------
def associate_tracks_to_dets(tracks, det_positions, det_valid):
    """
    Class-aware + 3D distance association (Hungarian if scipy available).
    Penalize class mismatch strongly to avoid ID swaps.
    """
    if len(tracks) == 0 or len(det_positions) == 0:
        return [], list(range(len(tracks))), list(range(len(det_positions)))

    cost = np.zeros((len(tracks), len(det_positions)), dtype=np.float32)

    for i, t in enumerate(tracks):
        for j, p in enumerate(det_positions):
            d3 = euclidean(t.pos3d, p)
            # big penalty if class mismatch
            if t.class_id != det_valid[j]["class_id"]:
                d3 += 50.0
            cost[i, j] = d3

    # Hungarian if available
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        matches, used_t, used_d = [], set(), set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= MAX_ASSOCIATION_DIST_M:
                matches.append((i, j))
                used_t.add(i)
                used_d.add(j)
    except Exception:
        # Greedy fallback
        matches, used_t, used_d = [], set(), set()
        tmp = cost.copy()
        while True:
            i, j = np.unravel_index(np.argmin(tmp), tmp.shape)
            m = tmp[i, j]
            if m > MAX_ASSOCIATION_DIST_M or np.isinf(m):
                break
            matches.append((i, j))
            used_t.add(i)
            used_d.add(j)
            tmp[i, :] = np.inf
            tmp[:, j] = np.inf
            if np.isinf(tmp).all():
                break

    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_dets   = [j for j in range(len(det_positions)) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets


# ----------------------- Main pipeline -----------------------
def process_sequence(seq_name, f, cx, cy):
    print(f"\n=== 3D TRACKING for {seq_name} ===")

    depth_dir = os.path.join(DEPTH_ROOT, seq_name, "depth")
    det_dir   = os.path.join(DETECTION_ROOT, seq_name, "data")

    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    det_files   = sorted(glob.glob(os.path.join(det_dir, "*_detections.npy")))

    if not depth_files or not det_files:
        print(f"[{seq_name}] Missing depth or detection outputs.")
        return

    n = min(len(depth_files), len(det_files))
    depth_files = depth_files[:n]
    det_files   = det_files[:n]

    out_seq = os.path.join(OUT_ROOT, seq_name)
    os.makedirs(out_seq, exist_ok=True)

    left_vis_dir = find_vis_left_dir(seq_name) if SAVE_VIS else None
    vis_out = os.path.join(out_seq, "visualizations")
    if SAVE_VIS and left_vis_dir:
        os.makedirs(vis_out, exist_ok=True)
        print(f"[{seq_name}] Visualizing on: {left_vis_dir}")

    tracks = []
    all_frames_output = []

    for frame_idx, (df, rf) in enumerate(zip(depth_files, det_files)):
        depth = np.load(df)
        dets  = load_detections(rf)

        # --- FIX 1: NMS to remove duplicate detections ---
        dets = nms_detections(dets, NMS_IOU_THRESH)

        # Lift to 3D
        det_positions, det_valid = [], []
        for det in dets:
            pos3d = detection_to_3d(det, depth, f, cx, cy)
            if pos3d is not None:
                det_positions.append(pos3d)
                det_valid.append(det)

        # Predict all tracks
        for t in tracks:
            t.predict()

        # Associate
        matches, _, unmatched_dets = associate_tracks_to_dets(tracks, det_positions, det_valid)

        # Update matched
        for ti, di in matches:
            tracks[ti].update(det_valid[di], det_positions[di])

        # New tracks
        for di in unmatched_dets:
            tracks.append(Track3D(det_valid[di], det_positions[di]))

        # Prune dead
        tracks = [t for t in tracks if not t.is_dead()]

        # Frame output
        frame_out = []
        for t in tracks:
            frame_out.append({
                "frame": frame_idx,
                "track_id": t.id,
                "class_name": t.class_name,
                "pos3d": [float(x) for x in t.pos3d],
                "bbox2d": t.last_bbox,
                "confidence": float(t.last_conf),
                "confirmed": t.is_confirmed(),
                "time_since_update": t.time_since_update
            })
        all_frames_output.append(frame_out)

        # Visualization (one box per TRACK)
        if SAVE_VIS and left_vis_dir:
            base = os.path.splitext(os.path.basename(df))[0]
            cand1 = os.path.join(left_vis_dir, base + ".png")
            cand2 = os.path.join(left_vis_dir, f"{int(base):010d}.png") if base.isdigit() else cand1
            img_path = cand1 if os.path.isfile(cand1) else cand2

            img = cv2.imread(img_path)
            if img is not None:
                for o in frame_out:
                    # optional: only show confirmed tracks
                    # if not o["confirmed"]: continue

                    x1, y1, x2, y2 = o["bbox2d"]
                    color = (0, 255, 0)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = f'ID {o["track_id"]} {o["class_name"]} Z={o["pos3d"][2]:.1f}m'
                    cv2.putText(img, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                out_img = os.path.join(vis_out, base + "_tracked.png")
                cv2.imwrite(out_img, img)

        if frame_idx % 25 == 0:
            print(f"[{seq_name}] Frame {frame_idx}/{n} | Active tracks: {len(tracks)}")

    # Save outputs
    with open(os.path.join(out_seq, "tracks_per_frame.json"), "w") as fjson:
        json.dump(all_frames_output, fjson, indent=2)

    tracklets = {t.id: t.history for t in tracks}
    with open(os.path.join(out_seq, "final_tracklets.json"), "w") as fjson:
        json.dump(tracklets, fjson, indent=2)

    print(f"[{seq_name}]  Saved 3D tracking results to {out_seq}")


def main():
    if not os.path.isfile(CALIB_NPZ):
        raise FileNotFoundError(f"Calibration file missing: {CALIB_NPZ}")

    f, cx, cy = load_calibration(CALIB_NPZ)
    print(f"Loaded calib: f={f:.2f}px, cx={cx:.2f}, cy={cy:.2f}")

    seqs = sorted([d for d in os.listdir(DEPTH_ROOT) if d.startswith("seq_")])
    if not seqs:
        raise RuntimeError(f"No sequences found in {DEPTH_ROOT}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    for s in seqs:
        process_sequence(s, f, cx, cy)

    print("\n3D tracking done for all sequences.")


if __name__ == "__main__":
    main()
