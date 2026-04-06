"""
Football field detection and yard-line based ground plane estimation.

Detects the green field surface and white yard-line hash marks to:
1. Estimate metric scale from yard line spacing (replaces ZoeDepth)
2. Estimate gravity/world alignment from the field plane (replaces SPEC)
3. Provide a field-textured ground mesh for visualization

Hash marks are 1 yard (0.9144 m) apart.
"""

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Standard football field dimensions in meters
YARD_IN_METERS = 0.9144
FIELD_LENGTH_YARDS = 100  # between end zones
FIELD_WIDTH_YARDS = 53.33
YARD_LINE_SPACING_YARDS = 1  # hash marks every 1 yard


# ---------------------------------------------------------------------------
# Low-level detection helpers
# ---------------------------------------------------------------------------

def detect_field_mask(image, hsv_lower=(30, 40, 40), hsv_upper=(85, 255, 255),
                      min_area_ratio=0.05):
    """
    Detect the green football field region using HSV color segmentation.

    :param image: BGR image (H, W, 3)
    :param hsv_lower: lower HSV bound for green
    :param hsv_upper: upper HSV bound for green
    :param min_area_ratio: minimum fraction of image area for valid field
    :returns: binary mask (H, W uint8 0/255), bool indicating if field was found
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask, False

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / (image.shape[0] * image.shape[1])
    if area_ratio < min_area_ratio:
        return mask, False

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    return clean_mask, True


def detect_yard_lines(image, field_mask, min_line_length=50, max_line_gap=20,
                      canny_low=50, canny_high=150, hough_threshold=80):
    """
    Detect white yard lines within the field region.

    :returns: (N, 4) array of line segments [x1,y1,x2,y2], dominant angle (rad)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    field_gray = cv2.bitwise_and(gray, gray, mask=field_mask)

    # White markings: bright pixels on green field
    _, white_mask = cv2.threshold(field_gray, 180, 255, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_and(white_mask, field_mask)

    edges = cv2.Canny(white_mask, canny_low, canny_high)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=hough_threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    if lines is None:
        return np.zeros((0, 4), dtype=np.float64), None

    line_segments = lines.reshape(-1, 4).astype(np.float64)

    # Angle of every segment, normalized to [0, pi)
    angles = np.arctan2(line_segments[:, 3] - line_segments[:, 1],
                        line_segments[:, 2] - line_segments[:, 0])
    angles[angles < 0] += np.pi

    # Find dominant direction via histogram (coarse)
    hist, bin_edges = np.histogram(angles, bins=36, range=(0, np.pi))
    dominant_bin = np.argmax(hist)
    coarse_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2

    # Keep lines within 15 deg of dominant
    tol = np.radians(15)
    diff = np.abs(angles - coarse_angle)
    aligned = (diff < tol) | (np.abs(diff - np.pi) < tol)

    # Refine: length-weighted mean angle of aligned segments
    aligned_segs = line_segments[aligned]
    aligned_angles = angles[aligned]
    lengths = np.sqrt((aligned_segs[:, 2] - aligned_segs[:, 0]) ** 2 +
                      (aligned_segs[:, 3] - aligned_segs[:, 1]) ** 2)
    # Use circular mean via unit vectors to avoid wrap-around issues
    dominant_angle = float(np.arctan2(
        np.sum(lengths * np.sin(2 * aligned_angles)),
        np.sum(lengths * np.cos(2 * aligned_angles))
    ) / 2)
    if dominant_angle < 0:
        dominant_angle += np.pi

    return aligned_segs, dominant_angle


def cluster_yard_lines(lines, dominant_angle, min_gap_px=10):
    """
    Cluster detected line segments into distinct yard lines.

    Projects midpoints onto the perpendicular axis and groups segments
    that fall within *min_gap_px* of each other.

    :returns: sorted 1-D array of perpendicular-axis positions (one per
              distinct yard line), perpendicular direction unit vector (2,)
    """
    if len(lines) < 2:
        return np.array([]), None

    perp_angle = dominant_angle + np.pi / 2
    perp_dir = np.array([np.cos(perp_angle), np.sin(perp_angle)])

    midpoints = (lines[:, :2] + lines[:, 2:]) / 2.0   # (N, 2)
    projections = midpoints @ perp_dir                  # (N,)
    projections_sorted = np.sort(projections)

    # Merge projections that belong to the same physical line
    clusters = [projections_sorted[0]]
    for p in projections_sorted[1:]:
        if p - clusters[-1] > min_gap_px:
            clusters.append(p)
        else:
            # Running mean
            clusters[-1] = (clusters[-1] + p) / 2.0

    return np.array(clusters), perp_dir


def compute_yard_line_spacing_pixels(lines, dominant_angle):
    """
    Median pixel spacing between adjacent detected yard lines.

    :returns: spacing in pixels, or None if fewer than 2 distinct lines
    """
    positions, _ = cluster_yard_lines(lines, dominant_angle)
    if len(positions) < 2:
        return None

    gaps = np.diff(positions)
    # Expect roughly uniform gaps; take the median
    median_gap = np.median(gaps)

    # Filter to gaps close to the median (within 50 %) to remove outliers
    near = gaps[(gaps > median_gap * 0.5) & (gaps < median_gap * 1.5)]
    return float(np.median(near)) if len(near) > 0 else float(median_gap)


# ---------------------------------------------------------------------------
# Per-frame field analysis
# ---------------------------------------------------------------------------

def estimate_field_scale(image, field_mask=None):
    """
    Per-frame: detect field, detect yard lines, compute pixels-per-yard.

    :returns: dict with keys pixels_per_yard, yard_line_angle, field_mask,
              yard_lines, line_positions, valid
    """
    result = dict(pixels_per_yard=None, yard_line_angle=None,
                  field_mask=None, yard_lines=np.zeros((0, 4)),
                  line_positions=np.array([]), valid=False)

    if field_mask is None:
        field_mask, found = detect_field_mask(image)
        if not found:
            return result
    result['field_mask'] = field_mask

    lines, dominant_angle = detect_yard_lines(image, field_mask)
    if dominant_angle is None or len(lines) < 2:
        return result
    result['yard_lines'] = lines
    result['yard_line_angle'] = dominant_angle

    positions, _ = cluster_yard_lines(lines, dominant_angle)
    result['line_positions'] = positions

    spacing = compute_yard_line_spacing_pixels(lines, dominant_angle)
    if spacing is None:
        return result

    # Each gap = YARD_LINE_SPACING_YARDS yards
    result['pixels_per_yard'] = spacing / YARD_LINE_SPACING_YARDS
    result['valid'] = True
    return result


# ---------------------------------------------------------------------------
# SLAM-integrated metric scale  (replaces ZoeDepth in run_metric_slam)
# ---------------------------------------------------------------------------

def est_scale_from_field(slam_depth, image, calib, human_mask=None):
    """
    Estimate the metric scale factor for a single SLAM keyframe by
    back-projecting pairs of adjacent yard lines into 3-D using the
    SLAM depth map and comparing their 3-D distance to the known
    physical distance (YARD_LINE_SPACING_YARDS yards).

    This replaces est_scale_hybrid / ZoeDepth for football-field videos.

    :param slam_depth: (H_slam, W_slam) depth from SLAM (1/disparity)
    :param image: BGR image at original resolution
    :param calib: [fx, fy, cx, cy] camera intrinsics at SLAM resolution
    :param human_mask: optional (H_orig, W_orig) mask, >0 where humans are
    :returns: scale float  (metric_depth = slam_depth * scale), or None
    """
    field_mask, found = detect_field_mask(image)
    if not found:
        return None

    # Exclude human pixels from the field mask
    if human_mask is not None:
        hm = cv2.resize(human_mask.astype(np.uint8),
                         (field_mask.shape[1], field_mask.shape[0]))
        field_mask = cv2.bitwise_and(field_mask, cv2.bitwise_not(hm * 255))

    lines, dominant_angle = detect_yard_lines(image, field_mask)
    positions, perp_dir = cluster_yard_lines(lines, dominant_angle)
    if len(positions) < 2 or perp_dir is None:
        return None

    fx, fy, cx, cy = calib[:4]
    H_slam, W_slam = slam_depth.shape
    H_img, W_img = image.shape[:2]

    # We will back-project the centre of each yard-line cluster into 3-D
    # using SLAM depth, then measure the 3-D gap between adjacent lines.
    line_along = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])

    scales = []
    for i in range(len(positions) - 1):
        p0 = positions[i]
        p1 = positions[i + 1]

        # Representative image pixel for each line (intersection of
        # perpendicular projection position and the line direction, taken
        # at the centre of the field mask along the line direction).
        for pos, label in [(p0, 'a'), (p1, 'b')]:
            # pixel = perp_dir * pos  + line_along * t  → pick t at image centre
            # We need a concrete pixel; use the midpoint of line segments in
            # the cluster.  Approximate: take the mean midpoint of segments
            # whose projection is near *pos*.
            mids = (lines[:, :2] + lines[:, 2:]) / 2.0
            proj = mids @ perp_dir
            close = np.abs(proj - pos) < 15  # pixels
            if close.sum() == 0:
                break
            px, py = mids[close].mean(axis=0)

            # Map from original-image coords to SLAM-depth coords
            sx = W_slam / W_img
            sy = H_slam / H_img
            u_slam = int(np.clip(px * sx, 0, W_slam - 1))
            v_slam = int(np.clip(py * sy, 0, H_slam - 1))

            # Sample a small patch of SLAM depth for robustness
            r = 3
            patch = slam_depth[max(0, v_slam - r):v_slam + r + 1,
                               max(0, u_slam - r):u_slam + r + 1]
            patch = patch[(patch > 0) & np.isfinite(patch)]
            if len(patch) == 0:
                break
            z = float(np.median(patch))

            # Back-project to camera 3-D
            X = (px * sx - cx) / fx * z
            Y = (py * sy - cy) / fy * z
            if label == 'a':
                pt_a = np.array([X, Y, z])
            else:
                pt_b = np.array([X, Y, z])
        else:
            # Both points computed successfully
            dist_slam = np.linalg.norm(pt_b - pt_a)
            if dist_slam < 1e-6:
                continue
            dist_metric = YARD_LINE_SPACING_YARDS * YARD_IN_METERS
            scales.append(dist_metric / dist_slam)

    if len(scales) == 0:
        return None

    return float(np.median(scales))


# ---------------------------------------------------------------------------
# Field-based world alignment  (replaces SPEC gravity in align_cam_to_world)
# ---------------------------------------------------------------------------

def get_field_ground_points(image, field_mask, slam_depth, calib):
    """
    Back-project field pixels to 3-D in camera frame using SLAM depth.

    :param image: BGR image (H_img, W_img, 3)
    :param field_mask: (H_img, W_img) uint8 mask
    :param slam_depth: (H_slam, W_slam) depth map
    :param calib: [fx, fy, cx, cy] at SLAM resolution
    :returns: (N, 3) float32 ndarray of 3-D points in camera frame
    """
    fx, fy, cx, cy = calib[:4]
    H_slam, W_slam = slam_depth.shape
    H_img, W_img = image.shape[:2]

    ys_img, xs_img = np.where(field_mask > 0)
    if len(xs_img) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Subsample for speed
    if len(xs_img) > 5000:
        idx = np.random.choice(len(xs_img), 5000, replace=False)
        xs_img, ys_img = xs_img[idx], ys_img[idx]

    # Map to SLAM depth coords
    sx = W_slam / W_img
    sy = H_slam / H_img
    us = (xs_img * sx).astype(int).clip(0, W_slam - 1)
    vs = (ys_img * sy).astype(int).clip(0, H_slam - 1)

    z = slam_depth[vs, us].astype(np.float32)
    valid = (z > 0) & np.isfinite(z)
    us, vs, z = us[valid], vs[valid], z[valid]

    X = (us.astype(np.float32) - cx) / fx * z
    Y = (vs.astype(np.float32) - cy) / fy * z
    return np.stack([X, Y, z], axis=-1)


def align_cam_to_field(imgfiles, cam_R, cam_T, calib, droid_disps, droid_tstamp,
                       human_masks=None, sample_frames=10):
    """
    Compute world alignment from the football field plane instead of SPEC.

    1. Sample keyframes, detect field, back-project field pixels using SLAM depth.
    2. Fit a plane via SVD → field normal = gravity direction.
    3. Compute yard-line direction in 3-D → consistent world-X/Z axes.
    4. Build R_wc that maps camera frame to a Y-up world frame aligned to the field.

    :param imgfiles: list of all image paths (full sequence)
    :param cam_R: (T, 3, 3) SLAM camera rotations
    :param cam_T: (T, 3)   SLAM camera translations (already metric-scaled)
    :param calib: [fx, fy, cx, cy]
    :param droid_disps: (K, H, W) upsampled disparity maps at keyframes
    :param droid_tstamp: (K,) frame indices for each keyframe
    :param human_masks: optional (T, H_orig, W_orig) tensor of human masks
    :param sample_frames: how many keyframes to use
    :returns: world_cam_R (T, 3, 3), world_cam_T (T, 3), field_normal (3,)
    """
    from lib.vis.traj import align_a2b

    K = len(droid_tstamp)
    indices = np.linspace(0, K - 1, min(sample_frames, K), dtype=int)

    all_pts = []  # 3-D field points accumulated across keyframes (in first-cam frame)
    yard_dirs_3d = []  # yard-line direction vectors in camera 3-D

    for ki in indices:
        t = int(droid_tstamp[ki])
        img = cv2.imread(imgfiles[t])
        if img is None:
            continue

        disp = droid_disps[ki]
        slam_depth = np.where(disp > 0, 1.0 / disp, 0.0)

        field_mask, found = detect_field_mask(img)
        if not found:
            continue

        # Exclude humans from field mask
        if human_masks is not None:
            hm = human_masks[t].numpy() if torch.is_tensor(human_masks[t]) else human_masks[t]
            hm_resized = cv2.resize(hm.astype(np.uint8),
                                     (field_mask.shape[1], field_mask.shape[0]))
            field_mask = cv2.bitwise_and(field_mask, cv2.bitwise_not(hm_resized * 255))

        pts_cam = get_field_ground_points(img, field_mask, slam_depth, calib)
        if pts_cam.shape[0] < 50:
            continue

        # Transform to the coordinate frame of the first camera (world = cam_0)
        R_i = cam_R[t].numpy() if torch.is_tensor(cam_R) else cam_R[t]
        T_i = cam_T[t].numpy() if torch.is_tensor(cam_T) else cam_T[t]
        pts_world = (R_i @ pts_cam.T).T + T_i  # (N, 3)
        all_pts.append(pts_world)

        # Yard-line direction in image → 3-D
        lines, dominant_angle = detect_yard_lines(img, field_mask)
        if dominant_angle is not None and len(lines) >= 2:
            # Line direction in image pixels
            ld_img = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
            # Pick two points along a line, back-project with mean field depth
            mean_z = float(np.median(pts_cam[:, 2]))
            H_slam, W_slam = slam_depth.shape
            H_img, W_img = img.shape[:2]
            sx, sy = W_slam / W_img, H_slam / H_img
            # Use image centre as reference
            cx_img, cy_img = W_img / 2, H_img / 2
            p0_img = np.array([cx_img, cy_img])
            p1_img = p0_img + ld_img * 100  # 100 px along line

            fx, fy, cx_s, cy_s = calib[:4]
            def backproj(px, py):
                u = px * sx
                v = py * sy
                X = (u - cx_s) / fx * mean_z
                Y = (v - cy_s) / fy * mean_z
                return np.array([X, Y, mean_z])

            pt0 = backproj(p0_img[0], p0_img[1])
            pt1 = backproj(p1_img[0], p1_img[1])
            d = pt1 - pt0
            d = d / (np.linalg.norm(d) + 1e-8)
            # Rotate to world frame
            d_world = R_i @ d
            yard_dirs_3d.append(d_world)

    # -- Fallback: if field not found, fall back to SPEC --
    if len(all_pts) < 2:
        print('  Field alignment: not enough field points, falling back to SPEC.')
        from lib.camera.est_gravity import align_cam_to_world as spec_align
        return spec_align(imgfiles[0], cam_R, cam_T)

    # -- Fit ground plane via SVD --
    all_pts = np.concatenate(all_pts, axis=0)
    if all_pts.shape[0] > 15000:
        idx = np.random.choice(all_pts.shape[0], 15000, replace=False)
        all_pts = all_pts[idx]

    mean = all_pts.mean(axis=0, keepdims=True)
    _, S, Vh = np.linalg.svd(all_pts - mean)
    normal = Vh[-1]  # smallest singular value = plane normal

    # Orient normal upward: for a camera looking at the ground the field
    # points' Y values (in camera convention, Y points down) should have
    # the normal pointing away from the camera centre → negative-Y in the
    # first camera's frame usually.  We pick the direction that makes
    # "up" point away from the mean field point relative to camera 0.
    cam0_pos = cam_T[0].numpy() if torch.is_tensor(cam_T) else cam_T[0]
    to_cam = cam0_pos - mean.squeeze()
    if np.dot(normal, to_cam) < 0:
        normal = -normal

    normal_t = torch.from_numpy(normal).float()
    normal_t = normal_t / normal_t.norm()
    yup = torch.tensor([0.0, 1.0, 0.0])
    R_field = align_a2b(normal_t, yup)  # (3, 3)

    # Optional: align one horizontal axis to the yard-line direction
    if len(yard_dirs_3d) >= 1:
        yd = np.mean(yard_dirs_3d, axis=0)
        yd = torch.from_numpy(yd).float()
        # Project onto horizontal plane after R_field rotation
        yd_rot = R_field @ yd
        yd_rot[1] = 0  # zero out vertical component
        if yd_rot.norm() > 1e-6:
            yd_rot = yd_rot / yd_rot.norm()
            # Build a Y-axis rotation that aligns yd_rot with world-X
            xaxis = torch.tensor([1.0, 0.0, 0.0])
            cos_a = torch.dot(yd_rot, xaxis).clamp(-1, 1)
            sin_a = torch.cross(yd_rot, xaxis)[1]  # y-component of cross
            R_yaw = torch.tensor([[ cos_a, 0, sin_a],
                                  [ 0,     1, 0    ],
                                  [-sin_a, 0, cos_a]]).float()
            R_field = R_yaw @ R_field

    # Apply to full trajectory
    if torch.is_tensor(cam_R):
        world_cam_R = torch.einsum('ij,bjk->bik', R_field, cam_R)
        world_cam_T = torch.einsum('ij,bj->bi', R_field, cam_T)
    else:
        R_np = R_field.numpy()
        world_cam_R = torch.from_numpy(np.einsum('ij,bjk->bik', R_np, cam_R)).float()
        world_cam_T = torch.from_numpy(np.einsum('ij,bj->bi', R_np, cam_T)).float()

    return world_cam_R, world_cam_T, R_field


# ---------------------------------------------------------------------------
# Top-level: field-aware metric SLAM  (replaces run_metric_slam for field videos)
# ---------------------------------------------------------------------------

def run_field_metric_slam(img_folder, masks=None, calib=None, is_static=False):
    """
    Drop-in replacement for run_metric_slam that uses yard-line spacing
    for metric scale instead of ZoeDepth, and field-plane normal for
    world alignment instead of SPEC.

    :returns: cam_R (T,3,3), cam_T (T,3) in metric scale,
              world_cam_R (T,3,3), world_cam_T (T,3) in field-aligned world frame,
              droid_disps (K,H,W) keyframe disparities,
              droid_tstamp (K,) keyframe timestamps
    """
    import sys
    sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
    sys.path.insert(0, 'thirdparty/DROID-SLAM')
    from glob import glob

    from lib.camera.masked_droid_slam import run_slam
    from lib.camera.slam_utils import est_calib, get_dimention
    from lib.utils.rotation_conversions import quaternion_to_matrix

    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    # --- Static camera shortcut ---
    if is_static:
        T = len(imgfiles)
        cam_r = torch.eye(3).expand(T, 3, 3)
        cam_t = torch.zeros(T, 3)
        return cam_r, cam_t, cam_r.clone(), cam_t.clone(), None, None

    # --- Run masked DROID-SLAM (same as original) ---
    if calib is None:
        calib = est_calib(img_folder)

    droid, traj = run_slam(img_folder, masks=masks, calib=calib)
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    del droid
    torch.cuda.empty_cache()

    # Convert SLAM trajectory to R, T (still in relative SLAM scale)
    slam_cam_t = torch.tensor(traj[:, :3])
    slam_cam_q = torch.tensor(traj[:, 3:])
    slam_cam_r = quaternion_to_matrix(slam_cam_q[:, [3, 0, 1, 2]])

    # --- Estimate metric scale from yard lines ---
    print('  Estimating metric scale from yard-line spacing ...')
    H_slam, W_slam = get_dimention(img_folder)
    H_img, W_img = cv2.imread(imgfiles[0]).shape[:2]
    # Intrinsics scaled to SLAM resolution
    fx, fy, cx, cy = calib[:4]
    sx_slam = W_slam / W_img
    sy_slam = H_slam / H_img
    calib_slam = [fx * sx_slam, fy * sy_slam, cx * sx_slam, cy * sy_slam]

    scales = []
    for i in tqdm(range(len(tstamp)), desc='Field scale'):
        t = tstamp[i]
        img = cv2.imread(imgfiles[t])
        if img is None:
            continue

        slam_depth = np.where(disps[i] > 0, 1.0 / disps[i], 0.0)

        hm = None
        if masks is not None:
            hm = masks[t].numpy() if torch.is_tensor(masks[t]) else masks[t]

        s = est_scale_from_field(slam_depth, img, calib_slam, human_mask=hm)
        if s is not None and 0.01 < s < 1000:
            scales.append(s)

    if len(scales) == 0:
        print('  WARNING: yard-line scale estimation failed on all keyframes.')
        print('           Falling back to ZoeDepth scale estimation.')
        from lib.camera.masked_droid_slam import run_metric_slam
        cam_r, cam_t = run_metric_slam(img_folder, masks=masks, calib=calib,
                                        is_static=is_static)
        # Fall back to SPEC for alignment too
        from lib.camera.est_gravity import align_cam_to_world
        wd_r, wd_t, _ = align_cam_to_world(imgfiles[0], cam_r, cam_t)
        return cam_r, cam_t, wd_r, wd_t, disps, tstamp

    scale = float(np.median(scales))
    print(f'  Field metric scale: {scale:.4f}  ({len(scales)}/{len(tstamp)} keyframes)')

    cam_t = slam_cam_t * scale
    cam_r = slam_cam_r

    # --- World alignment from field plane ---
    print('  Aligning world frame to football field plane ...')
    world_cam_R, world_cam_T, _ = align_cam_to_field(
        imgfiles, cam_r, cam_t, calib_slam,
        disps, tstamp, human_masks=masks, sample_frames=15
    )

    return cam_r, cam_t, world_cam_R, world_cam_T, disps, tstamp
