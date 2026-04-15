"""
Football field detection and yard-line based ground plane estimation.

Detects the green field surface and the long white yard-line stripes to:
1. Estimate metric scale from yard line spacing (replaces ZoeDepth)
2. Estimate gravity/world alignment from the field plane (replaces SPEC)
3. Provide a field-textured ground mesh for visualization

Major yard lines (the long stripes that cross the full field width) are 5 yards
(4.572 m) apart.  Hash marks (short tick marks in the middle third of the field)
are filtered out so only the 5-yard stripes are used for scale estimation.
"""

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Standard football field dimensions in meters
YARD_IN_METERS = 0.9144
FIELD_LENGTH_YARDS = 100  # between end zones
FIELD_WIDTH_YARDS = 53.33
YARD_LINE_SPACING_YARDS = 5  # major yard lines every 5 yards

# Training hoops placed on the field for drill work
HOOP_DIAMETER_FEET = 12
HOOP_DIAMETER_METERS = HOOP_DIAMETER_FEET * 0.3048  # 3.6576 m


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


def filter_long_yard_lines(lines, dominant_angle, positions, perp_dir, image_width,
                            min_span_ratio=0.50, max_lines=5, min_spacing_px=110,
                            min_coverage_ratio=0.35):
    """
    Keep only yard-line clusters that look like continuous stripes, not
    scattered hash marks, up to *max_lines* results.

    Three filters are applied per cluster:
    1. **Span** — total extent along the line direction must be ≥
       *min_span_ratio* × image_width (rejects short marks).
    2. **Coverage** — sum of individual segment lengths / total span must be ≥
       *min_coverage_ratio* (rejects hash marks whose combined span is large
       but whose segments are separated by wide gaps).
    3. **Spacing** — each kept line must be ≥ *min_spacing_px* from every
       other kept line on the perpendicular axis.

    Candidates that pass filters 1 & 2 are ranked by coverage ratio
    (most continuous first), then selected greedily with the spacing constraint.

    :param lines: (N, 4) line segments from detect_yard_lines
    :param dominant_angle: dominant line angle in radians
    :param positions: 1-D cluster positions from cluster_yard_lines
    :param perp_dir: perpendicular unit vector from cluster_yard_lines
    :param image_width: width of the source image in pixels
    :param min_span_ratio: minimum span / image_width (default 0.50)
    :param max_lines: maximum lines to return (default 5)
    :param min_spacing_px: minimum distance between kept lines in px (default 110)
    :param min_coverage_ratio: minimum covered_length / span (default 0.35)
    :returns: filtered 1-D array of cluster positions (sorted by perp position)
    """
    if len(positions) == 0 or perp_dir is None or len(lines) == 0:
        return positions

    line_dir = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
    mids = (lines[:, :2] + lines[:, 2:]) / 2.0
    proj_perp = mids @ perp_dir

    min_span = min_span_ratio * image_width
    scored = []  # (coverage_ratio, position)
    for pos in positions:
        mask = np.abs(proj_perp - pos) < 15
        if mask.sum() == 0:
            continue
        segs = lines[mask]

        # Total span: leftmost to rightmost projected endpoint
        all_pts = np.vstack([segs[:, :2], segs[:, 2:]])  # (2N, 2)
        proj_along = all_pts @ line_dir
        span = float(proj_along.max() - proj_along.min())
        if span < min_span:
            continue

        # Coverage: sum of each segment's projected length
        p0 = segs[:, :2] @ line_dir
        p1 = segs[:, 2:] @ line_dir
        covered = float(np.abs(p1 - p0).sum())
        coverage_ratio = covered / span
        if coverage_ratio < min_coverage_ratio:
            continue

        scored.append((coverage_ratio, pos))

    if not scored:
        return np.array([])

    # Greedy selection: most-continuous lines first, enforce min spacing
    scored.sort(key=lambda x: -x[0])
    kept_pos = []
    for _, pos in scored:
        if all(abs(pos - k) >= min_spacing_px for k in kept_pos):
            kept_pos.append(pos)
        if len(kept_pos) == max_lines:
            break

    return np.array(sorted(kept_pos))


def compute_yard_line_spacing_pixels(lines, dominant_angle, positions=None):
    """
    Median pixel spacing between adjacent detected yard lines.

    :param positions: pre-computed (and optionally pre-filtered) cluster positions;
                      if None, cluster_yard_lines is called internally.
    :returns: spacing in pixels, or None if fewer than 2 distinct lines
    """
    if positions is None:
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
# Hoop (circular marker) detection and scale estimation
# ---------------------------------------------------------------------------

def detect_hoops(image, field_mask=None, max_hoops=2,
                 min_contour_area=300, max_contour_area=60000,
                 min_aspect=0.08, min_semi_major=40,
                 overlap_thresh=0.8):
    """
    Detect red training hoops lying flat on the football field.

    Hoops laid flat appear as *ellipses* (foreshortened circles) rather than
    perfect circles.  Detection uses red HSV segmentation + contour fitting:

    1. Build a red-pixel mask (HSV hue 0-10 and 160-180).
    2. Find contours within the field region.
    3. Fit an ellipse to each large-enough red contour.
    4. Reject ellipses whose semi-major axis is < min_semi_major (removes
       small red objects like shirt logos).
    5. Apply distance-based NMS: if two ellipse centres are within
       overlap_thresh × semi_major of the larger ellipse, keep only the
       larger one (removes duplicate detections of the same hoop).
    6. Return the *max_hoops* largest surviving candidates.

    For a flat circle the major axis of the projected ellipse equals the
    true diameter (no foreshortening along that axis).

    :param image: BGR image (H, W, 3)
    :param field_mask: optional uint8 mask restricting detection to field region
    :param max_hoops: maximum detections to return (default 2)
    :param min_contour_area: minimum contour area in px² (filters tiny noise)
    :param max_contour_area: maximum contour area in px² (filters large blobs)
    :param min_aspect: minimum minor/major ratio; very flat shapes are noise
    :param min_semi_major: minimum semi-major axis in pixels; filters small
                           objects such as shirt logos (default 40 px)
    :param overlap_thresh: NMS distance threshold as a fraction of the larger
                           ellipse's semi-major axis (default 0.8)
    :returns: (N, 5) float32 array of
              [cx, cy, half_w, half_h, angle_deg], N ≤ max_hoops.
              (half_w, half_h, angle) match cv2.fitEllipse / cv2.ellipse order.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red wraps in HSV: low hue 0-10, high hue 160-180
    red_lo = cv2.inRange(hsv, np.array([0,   80, 60]), np.array([10,  255, 255]))
    red_hi = cv2.inRange(hsv, np.array([160, 80, 60]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_lo, red_hi)

    if field_mask is not None:
        red_mask = cv2.bitwise_and(red_mask, field_mask)

    # Close gaps in the ring, remove speckle
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area or area > max_contour_area:
            continue
        if len(cnt) < 5:  # fitEllipse needs >= 5 points
            continue

        (cx, cy), (w, h), angle = cv2.fitEllipse(cnt)
        half_w, half_h = w / 2.0, h / 2.0
        semi_major = max(half_w, half_h)

        if semi_major < min_semi_major:
            continue
        if (min(half_w, half_h) / semi_major) < min_aspect:
            continue

        # Store (cx, cy, half_w, half_h, angle) — axis order matches
        # cv2.fitEllipse output so cv2.ellipse can consume it directly.
        candidates.append((area, cx, cy, half_w, half_h, angle))

    if not candidates:
        return np.zeros((0, 5), dtype=np.float32)

    # Sort largest-first so NMS always keeps the bigger detection
    candidates.sort(key=lambda x: -x[0])

    # Distance-based NMS: suppress smaller ellipses that overlap a larger one
    kept = []
    for cand in candidates:
        _, cx_c, cy_c, hw_c, hh_c, _ = cand
        semi_c = max(hw_c, hh_c)
        suppressed = False
        for _, cx_k, cy_k, hw_k, hh_k, _ in kept:
            dist = np.hypot(cx_c - cx_k, cy_c - cy_k)
            if dist < overlap_thresh * max(semi_c, max(hw_k, hh_k)):
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
        if len(kept) == max_hoops:
            break

    return np.array([[cx, cy, hw, hh, ang]
                     for _, cx, cy, hw, hh, ang in kept],
                    dtype=np.float32)


def est_scale_from_hoops(slam_depth, image, calib, field_mask=None):
    """
    Estimate metric scale from detected circular training hoops of known
    diameter HOOP_DIAMETER_METERS (12 ft = 3.6576 m).

    For each detected circle the SLAM depth at the centre is sampled and
    the projected physical diameter at that depth is computed; the scale
    factor is derived as  scale = HOOP_DIAMETER_METERS / diameter_slam.

    :param slam_depth: (H_slam, W_slam) SLAM depth map (1/disparity)
    :param image: BGR image at original resolution
    :param calib: [fx, fy, cx, cy] at SLAM resolution
    :param field_mask: optional field mask at original-image resolution
    :returns: scale float or None
    """
    hoops = detect_hoops(image, field_mask)
    if len(hoops) == 0:
        return None

    fx, fy, *_ = calib[:4]
    H_slam, W_slam = slam_depth.shape
    H_img, W_img = image.shape[:2]
    sx = W_slam / W_img
    sy = H_slam / H_img
    f_avg = (fx + fy) / 2.0

    scales = []
    for hoop in hoops:
        cx_px, cy_px = hoop[0], hoop[1]
        # half_w / half_h are in fitEllipse axis order; true semi-major is the larger
        semi_major_px = max(hoop[2], hoop[3])
        u_slam = int(np.clip(cx_px * sx, 0, W_slam - 1))
        v_slam = int(np.clip(cy_px * sy, 0, H_slam - 1))

        # Sample a patch around the hoop centre for robust depth
        pr = max(3, int(semi_major_px * sx * 0.2))
        patch = slam_depth[
            max(0, v_slam - pr):v_slam + pr + 1,
            max(0, u_slam - pr):u_slam + pr + 1,
        ]
        patch = patch[(patch > 0) & np.isfinite(patch)]
        if len(patch) == 0:
            continue
        z = float(np.median(patch))

        # For a flat circle the major axis = true diameter (no foreshortening).
        # Convert major axis from image pixels to SLAM-depth pixels, then to
        # physical SLAM units via pinhole:  length = px_slam * z / f
        diameter_slam = (semi_major_px * 2.0 * sx) * z / f_avg
        if diameter_slam < 1e-6:
            continue

        scales.append(HOOP_DIAMETER_METERS / diameter_slam)

    if len(scales) == 0:
        return None
    return float(np.median(scales))


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

    positions, perp_dir = cluster_yard_lines(lines, dominant_angle)
    # Keep only the long 5-yard stripes; drop short hash marks
    positions = filter_long_yard_lines(lines, dominant_angle, positions, perp_dir,
                                        image.shape[1])
    result['line_positions'] = positions

    spacing = compute_yard_line_spacing_pixels(lines, dominant_angle, positions=positions)
    if spacing is None:
        return result

    # Each gap = YARD_LINE_SPACING_YARDS yards (5-yard major lines)
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
    # Keep only long 5-yard stripes
    positions = filter_long_yard_lines(lines, dominant_angle, positions, perp_dir,
                                        image.shape[1])
    if len(positions) < 2 or perp_dir is None:
        return None

    fx, fy, cx, cy = calib[:4]
    H_slam, W_slam = slam_depth.shape
    H_img, W_img = image.shape[:2]
    sx = W_slam / W_img
    sy = H_slam / H_img

    line_along = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
    mids = (lines[:, :2] + lines[:, 2:]) / 2.0
    proj_perp  = mids @ perp_dir
    proj_along = mids @ line_along

    def _backproject(px, py):
        """Back-project image pixel (px, py) using SLAM depth; returns 3-D point or None."""
        u = int(np.clip(px * sx, 0, W_slam - 1))
        v = int(np.clip(py * sy, 0, H_slam - 1))
        r = 3
        patch = slam_depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        patch = patch[(patch > 0) & np.isfinite(patch)]
        if len(patch) == 0:
            return None
        z = float(np.median(patch))
        X = (px * sx - cx) / fx * z
        Y = (py * sy - cy) / fy * z
        return np.array([X, Y, z])

    scales = []
    for i in range(len(positions) - 1):
        p0 = positions[i]
        p1 = positions[i + 1]

        close_a = np.abs(proj_perp - p0) < 15
        close_b = np.abs(proj_perp - p1) < 15
        if close_a.sum() == 0 or close_b.sum() == 0:
            continue

        # Sample both points at the same along-line coordinate so the 3-D
        # vector (pt_b - pt_a) is perpendicular to the yard lines with no
        # spurious lateral component.  Use the midpoint of the along-line
        # overlap region shared by both clusters; fall back to the mean of
        # both clusters' along-line extents if they don't overlap.
        a_along = proj_along[close_a]
        b_along = proj_along[close_b]
        t_lo = max(a_along.min(), b_along.min())
        t_hi = min(a_along.max(), b_along.max())
        t_ref = (t_lo + t_hi) / 2.0 if t_lo < t_hi \
                else (a_along.mean() + b_along.mean()) / 2.0

        # Pixel coordinates for each line at t_ref
        px_a = t_ref * line_along[0] + p0 * perp_dir[0]
        py_a = t_ref * line_along[1] + p0 * perp_dir[1]
        px_b = t_ref * line_along[0] + p1 * perp_dir[0]
        py_b = t_ref * line_along[1] + p1 * perp_dir[1]

        pt_a = _backproject(px_a, py_a)
        pt_b = _backproject(px_b, py_b)
        if pt_a is None or pt_b is None:
            continue

        # The distance between the two co-aligned points is the perpendicular
        # spacing between adjacent yard lines = YARD_LINE_SPACING_YARDS yards.
        dist_slam = np.linalg.norm(pt_b - pt_a)
        if dist_slam < 1e-6:
            continue
        dist_metric = YARD_LINE_SPACING_YARDS * YARD_IN_METERS
        scales.append(dist_metric / dist_slam)

    if len(scales) == 0:
        return None

    return float(np.median(scales))


# ---------------------------------------------------------------------------
# Pitch / roll estimation from yard-line perspective convergence
# ---------------------------------------------------------------------------

def estimate_pitch_from_yard_lines(image, calib):
    """
    Estimate camera pitch purely from the perspective convergence of detected
    yard lines — no SLAM depth required.

    Yard lines are equally-spaced (5 yards) parallel lines on the ground.
    Viewed from an elevated camera they converge toward a vanishing point V
    whose position encodes the camera's downward tilt:

        pitch = arctan( (c_perp - V) / f_perp )

    where c_perp is the principal-point coordinate projected onto the
    direction perpendicular to the yard lines, and f_perp is the effective
    focal length in that direction.

    The algorithm fits a 1-D projective transform

        p_k = (a·k + b) / (c·k + 1)   for k = 0, 1, …, N-1

    to the perpendicular-axis positions of N detected yard-line clusters
    (sorted near-to-far, i.e. decreasing p).  The vanishing point is V = a/c,
    which is correctly extrapolated even when V lies above the image.

    Note: optical-axis roll is intentionally not estimated here because
    cam_wrt_world expects SPEC-convention "roll" (heading about gravity Y),
    not camera roll.  SPEC supplies that value separately.

    :param image:  BGR image (H, W, 3)
    :param calib:  [fx, fy, cx, cy] at the image's resolution
    :returns: pitch_deg (float) or None on failure
    """
    fx, fy, cx, cy = calib[:4]

    field_mask, found = detect_field_mask(image)
    if not found:
        return None

    lines, dominant_angle = detect_yard_lines(image, field_mask)
    if lines is None or len(lines) == 0:
        return None

    # Cluster then filter to major yard lines (removes hash marks)
    positions, perp_dir = cluster_yard_lines(lines, dominant_angle)
    if perp_dir is None:
        return None
    positions = filter_long_yard_lines(lines, dominant_angle, positions, perp_dir,
                                       image.shape[1])
    if len(positions) < 3:
        return None

    # Sort near-to-far: for a forward-looking camera, near yard lines project
    # to larger perpendicular-axis values (lower in image), so sort descending.
    positions = np.sort(positions)[::-1].copy()   # descending = near → far
    N = len(positions)
    ks = np.arange(N, dtype=float)

    # Linearise the projective model:
    #   p_k · (c·k + 1) = a·k + b
    #   ⟹  a·k  −  c·p_k·k  +  b  =  p_k
    #   ⟹  [k, −p_k·k, 1] · [a, c, b]ᵀ = p_k
    A = np.column_stack([ks, -positions * ks, np.ones(N)])
    result, _, _, _ = np.linalg.lstsq(A, positions, rcond=None)
    a_coeff, c_coeff, _ = result

    if abs(c_coeff) < 1e-9:
        return None

    vp = a_coeff / c_coeff   # vanishing-point position on perpendicular axis

    # Principal-point and effective focal length projected onto perp_dir
    px, py = float(perp_dir[0]), float(perp_dir[1])
    c_perp = cx * px + cy * py
    # For a calibrated camera the effective focal length along perp_dir is:
    #   f_perp ≈ sqrt( (fx·px)² + (fy·py)² )
    f_perp = float(np.sqrt((fx * px) ** 2 + (fy * py) ** 2))
    if f_perp < 1.0:
        f_perp = float(fy)   # fallback for degenerate perp_dir

    pitch_rad = np.arctan2(c_perp - vp, f_perp)
    pitch_deg = float(np.degrees(pitch_rad))

    # Only return pitch — roll (optical-axis tilt) is not used here.
    # cam_wrt_world expects SPEC-convention "roll" = heading rotation about
    # the gravity Y-axis, which is completely different from the optical roll
    # visible in yard line tilt.  We let SPEC supply that value separately.
    return pitch_deg


def align_cam_via_yard_lines(imgfiles, cam_R, cam_T, calib, sample_frames=15):
    """
    World-align a SLAM trajectory using camera pitch inferred from the
    perspective convergence of yard lines, combined with SPEC for heading.

    SPEC's pitch estimate is unreliable for high-angle cameras, but its
    heading ("roll" in SPEC convention = rotation about gravity Y) is
    correct.  This function:
      1. Runs SPEC on the first frame to get the heading rotation.
      2. Estimates pitch from yard-line vanishing point (more accurate for
         high-angle / nearly-static football cameras).
      3. Rebuilds R_wc with the corrected pitch but SPEC's heading.

    Falls back to pure SPEC if too few yard lines are detected.

    :param imgfiles:      sorted list of image paths (full sequence)
    :param cam_R:         (T, 3, 3) SLAM camera rotations  (camera-to-world)
    :param cam_T:         (T, 3)   SLAM camera translations (metric-scaled)
    :param calib:         [fx, fy, cx, cy] at original image resolution
    :param sample_frames: how many frames to sample for robust pitch estimation
    :returns: world_cam_R (T, 3, 3), world_cam_T (T, 3), R_wc (3, 3)
    """
    from lib.camera.est_gravity import run_spec, cam_wrt_world

    # --- Step 1: SPEC for heading (spec_roll = rotation about gravity Y-axis) ---
    # spec_pitch and spec_roll are in radians.
    _, spec_pitch_rad, spec_roll_rad = run_spec(imgfiles[0])

    # --- Step 2: yard-line vanishing-point pitch (returned in degrees) ---
    T_total = len(imgfiles)
    indices = np.linspace(0, T_total - 1, min(sample_frames, T_total), dtype=int)

    pitches_deg = []
    for i in indices:
        img = cv2.imread(imgfiles[i])
        if img is None:
            continue
        p = estimate_pitch_from_yard_lines(img, calib)
        if p is not None:
            pitches_deg.append(p)

    if len(pitches_deg) < 3:
        print('  Yard-line pitch: too few detections, keeping SPEC pitch '
              f'({np.degrees(spec_pitch_rad):.1f}°).')
        effective_pitch_rad = spec_pitch_rad
    else:
        yard_pitch_deg = float(np.median(pitches_deg))
        print(f'  Yard-line pitch: {yard_pitch_deg:.1f}°  (n={len(pitches_deg)}, '
              f'SPEC was {np.degrees(spec_pitch_rad):.1f}°)')
        effective_pitch_rad = np.radians(yard_pitch_deg)

    # --- Step 3: R_wc with yard-line pitch + SPEC heading ---
    R_wc = cam_wrt_world(effective_pitch_rad, spec_roll_rad)

    if torch.is_tensor(cam_R):
        world_cam_R = torch.einsum('ij,bjk->bik', R_wc, cam_R)
        world_cam_T = torch.einsum('ij,bj->bi',   R_wc, cam_T)
    else:
        R_np = R_wc.numpy()
        world_cam_R = torch.from_numpy(
            np.einsum('ij,bjk->bik', R_np, cam_R)).float()
        world_cam_T = torch.from_numpy(
            np.einsum('ij,bj->bi',   R_np, cam_T)).float()

    return world_cam_R, world_cam_T, R_wc


# ---------------------------------------------------------------------------
# Field-based world alignment  (SLAM-depth plane fitting — legacy)
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

    # Shift the world origin to the field plane so the ground is at Y = 0.
    # After rotation, the fitted field plane passes through R_field @ mean at
    # some Y offset — without this shift the camera sits at the SLAM origin
    # (often near Y = 0) while the field is above or below it.
    field_center = torch.from_numpy(mean.squeeze()).float()
    field_y = (R_field @ field_center)[1]   # Y of the field plane in world frame
    world_cam_T[:, 1] -= field_y            # translate so field is at Y = 0

    return world_cam_R, world_cam_T, R_field


# ---------------------------------------------------------------------------
# Top-level: field-aware metric SLAM  (replaces run_metric_slam for field videos)
# ---------------------------------------------------------------------------

def run_field_metric_slam(img_folder, masks=None, calib=None, is_static=False,
                          use_hoops=False):
    """
    Drop-in replacement for run_metric_slam that uses yard-line spacing
    for metric scale instead of ZoeDepth, and field-plane normal for
    world alignment instead of SPEC.

    :param use_hoops: if True, also attempt hoop-based scale estimation on
                      each keyframe and pool those estimates with yard-line ones
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

        if use_hoops:
            s_hoop = est_scale_from_hoops(slam_depth, img, calib_slam)
            if s_hoop is not None and 0.01 < s_hoop < 1000:
                scales.append(s_hoop)

    if len(scales) == 0:
        print('  WARNING: yard-line/hoop scale estimation failed on all keyframes.')
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

    # --- World alignment from yard-line vanishing point ---
    # Uses image-only geometry: no SLAM depth needed, works for static/high-angle cameras.
    print('  Aligning world frame via yard-line vanishing point ...')
    world_cam_R, world_cam_T, R_wc = align_cam_via_yard_lines(
        imgfiles, cam_r, cam_t, calib, sample_frames=15
    )

    # --- Height shift: translate so the field surface sits at Y = 0 ---
    # The rotation above only re-orients the frame; the SLAM origin is camera 0's
    # position, so the ground is at negative Y.  Back-project a few keyframes of
    # field pixels (using metric-scale depth) to find where the ground plane is,
    # then shift world_cam_T so that plane moves to Y = 0.
    R_wc_np = R_wc.numpy() if torch.is_tensor(R_wc) else np.array(R_wc)
    ground_ys = []
    for ki in np.linspace(0, len(tstamp) - 1, min(6, len(tstamp)), dtype=int):
        t = int(tstamp[ki])
        img = cv2.imread(imgfiles[t])
        if img is None:
            continue
        # Metric-scale depth: disparity × scale
        slam_depth_m = np.where(disps[ki] > 0, scale / disps[ki], 0.0)
        field_mask, found = detect_field_mask(img)
        if not found:
            continue
        hm = None
        if masks is not None:
            hm = masks[t].numpy() if torch.is_tensor(masks[t]) else masks[t]
            if hm is not None:
                hm_rs = cv2.resize(hm.astype(np.uint8),
                                   (field_mask.shape[1], field_mask.shape[0]))
                field_mask = cv2.bitwise_and(field_mask,
                                             cv2.bitwise_not(hm_rs * 255))
        pts_cam = get_field_ground_points(img, field_mask, slam_depth_m, calib_slam)
        if pts_cam.shape[0] < 20:
            continue
        # Transform to SLAM world frame then to aligned world frame
        R_i = cam_r[t].numpy() if torch.is_tensor(cam_r[t]) else cam_r[t]
        T_i = cam_t[t].numpy() if torch.is_tensor(cam_t[t]) else cam_t[t]
        pts_slam = (R_i @ pts_cam.T).T + T_i
        pts_world = (R_wc_np @ pts_slam.T).T
        ground_ys.append(float(pts_world[:, 1].mean()))

    if ground_ys:
        ground_y = float(np.median(ground_ys))
        world_cam_T = world_cam_T.clone()
        world_cam_T[:, 1] -= ground_y
        print(f'  Height shift: {ground_y:+.2f} m  (field → Y=0, '
              f'camera ~{-ground_y:.1f} m above ground)')

    return cam_r, cam_t, world_cam_R, world_cam_T, disps, tstamp
