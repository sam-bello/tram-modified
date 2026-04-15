"""
Visualize football field detection: field mask, yard lines, and clustered line positions.

Usage:
    python scripts/visualize_field_detection.py --video "2022 NIC BARNO AMARE DL25_trimmed.mp4"
    python scripts/visualize_field_detection.py --video "2022 NIC BARNO AMARE DL25_trimmed.mp4" --frame 100
    python scripts/visualize_field_detection.py --video "2022 NIC BARNO AMARE DL25_trimmed.mp4" --all_frames
"""

import argparse
import os
import sys
import cv2
import numpy as np
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.pipeline.field_detection import (
    detect_field_mask, detect_yard_lines, cluster_yard_lines,
    filter_long_yard_lines, compute_yard_line_spacing_pixels,
    detect_hoops, YARD_IN_METERS, YARD_LINE_SPACING_YARDS, HOOP_DIAMETER_FEET
)


def draw_red_debug(image, field_mask=None, frame_idx=None):
    """
    Return a debug panel showing:
      Left  — red HSV mask (white = red pixels)
      Right — original image with every raw red contour drawn + area label

    This helps diagnose what the hoop detector actually sees before any
    ellipse fitting or size filtering is applied.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, np.array([0,   80, 60]), np.array([10,  255, 255]))
    red_hi = cv2.inRange(hsv, np.array([160, 80, 60]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_lo, red_hi)

    if field_mask is not None:
        red_mask = cv2.bitwise_and(red_mask, field_mask)

    # Morphological close+open (same as detect_hoops)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)
    red_mask_clean = cv2.morphologyEx(red_mask_clean, cv2.MORPH_OPEN, k_open)

    # Left panel: red mask as a colour image (red tint for visibility)
    mask_vis = np.zeros_like(image)
    mask_vis[red_mask > 0] = (0, 0, 180)        # raw red pixels in dark-red
    mask_vis[red_mask_clean > 0] = (0, 0, 255)  # after morphology in bright-red

    # Right panel: original image + all contours + area labels
    contour_vis = image.copy()
    contours, _ = cv2.findContours(red_mask_clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        cv2.drawContours(contour_vis, [cnt], -1, (0, 255, 255), 2)
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = cnt[0][0]
        label = f'#{i} {int(area)}px2'
        cv2.putText(contour_vis, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(contour_vis, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # If large enough for fitEllipse, show the fitted ellipse too
        if len(cnt) >= 5 and area >= 300:
            (ex, ey), (ew, eh), eangle = cv2.fitEllipse(cnt)
            hw, hh = int(ew / 2), int(eh / 2)
            if hw > 0 and hh > 0:
                cv2.ellipse(contour_vis, (int(ex), int(ey)), (hw, hh),
                            eangle, 0, 360, (255, 128, 0), 2)

    # Header labels
    H, W = image.shape[:2]
    for panel, title in [(mask_vis, 'RED MASK (raw=dark / morphed=bright)'),
                         (contour_vis, f'RED CONTOURS  ({len(contours)} total)')]:
        cv2.putText(panel, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(panel, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if frame_idx is not None:
            cv2.putText(panel, f'Frame {frame_idx}', (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(panel, f'Frame {frame_idx}', (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return np.hstack([mask_vis, contour_vis])


def draw_field_detection(image, frame_idx=None, hoop_mode=False):
    """
    Run field detection on a single image and return an annotated visualization.

    Returns: annotated BGR image, info dict
    """
    H, W = image.shape[:2]
    info = {}

    # 1) Detect field mask
    field_mask, found = detect_field_mask(image)
    info['field_found'] = found

    if not found:
        vis = image.copy()
        cv2.putText(vis, 'NO FIELD DETECTED', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if frame_idx is not None:
            cv2.putText(vis, f'Frame {frame_idx}', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return vis, info

    # 2) Detect yard lines
    lines, dominant_angle = detect_yard_lines(image, field_mask)
    info['num_raw_lines'] = len(lines)
    info['dominant_angle_deg'] = np.degrees(dominant_angle) if dominant_angle is not None else None

    # 3) Cluster into distinct yard lines, then keep only long 5-yard stripes
    positions, perp_dir = cluster_yard_lines(lines, dominant_angle) if dominant_angle is not None else (np.array([]), None)
    if dominant_angle is not None and perp_dir is not None:
        positions = filter_long_yard_lines(lines, dominant_angle, positions, perp_dir, W)
    info['num_clusters'] = len(positions)

    # 4) Compute spacing using pre-filtered positions
    spacing_px = compute_yard_line_spacing_pixels(lines, dominant_angle, positions=positions) if dominant_angle is not None and len(lines) >= 2 else None
    info['spacing_px'] = spacing_px
    if spacing_px is not None:
        info['pixels_per_yard'] = spacing_px / YARD_LINE_SPACING_YARDS
    else:
        info['pixels_per_yard'] = None

    # 5) Optionally detect hoops
    hoops = detect_hoops(image, field_mask) if hoop_mode else np.zeros((0, 5), dtype=np.float32)
    info['num_hoops'] = len(hoops)

    # --- Build visualization ---
    # Start with field mask overlay (green tint on field region)
    vis = image.copy()
    overlay = vis.copy()
    overlay[field_mask > 0] = (overlay[field_mask > 0] * 0.6 + np.array([0, 80, 0], dtype=np.uint8) * 0.4).astype(np.uint8)
    vis = overlay

    # Draw field mask contour
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # Draw raw detected line segments as dashed blue lines
    for seg in lines:
        x1, y1, x2, y2 = seg.astype(int)
        pt1 = np.array([x1, y1], dtype=np.float64)
        pt2 = np.array([x2, y2], dtype=np.float64)
        length = np.linalg.norm(pt2 - pt1)
        if length < 1:
            continue
        direction = (pt2 - pt1) / length
        dash_len = 8
        gap_len = 6
        d = 0.0
        while d < length:
            d_end = min(d + dash_len, length)
            p1 = (pt1 + direction * d).astype(int)
            p2 = (pt1 + direction * d_end).astype(int)
            cv2.line(vis, tuple(p1), tuple(p2), (255, 100, 0), 2)
            d += dash_len + gap_len

    # Draw clustered yard line positions as colored lines spanning the image
    if perp_dir is not None and len(positions) > 0 and dominant_angle is not None:
        line_dir = np.array([np.cos(dominant_angle), np.sin(dominant_angle)])
        colors = [
            (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
            (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
            (255, 128, 0), (0, 128, 255)
        ]

        for i, pos in enumerate(positions):
            # Reconstruct a point on this cluster line: perp_dir * pos
            # Then extend along line_dir to span the image
            center = perp_dir * pos
            pt1 = center - line_dir * max(W, H)
            pt2 = center + line_dir * max(W, H)
            color = colors[i % len(colors)]
            cv2.line(vis, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
                     color, 3)

        # Draw spacing annotations between adjacent clusters
        for i in range(len(positions) - 1):
            mid_pos = (positions[i] + positions[i + 1]) / 2
            center = perp_dir * mid_pos
            gap_px = positions[i + 1] - positions[i]
            label = f'{gap_px:.1f}px'
            cv2.putText(vis, label, (int(center[0]) + 5, int(center[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw detected hoops as ellipses (flat hoops appear foreshortened).
    # half_w / half_h are stored in cv2.fitEllipse axis order — pass directly
    # to cv2.ellipse so the orientation matches the detected contour exactly.
    for cx_px, cy_px, half_w, half_h, angle in hoops:
        cx_i, cy_i = int(cx_px), int(cy_px)
        axes = (max(1, int(half_w)), max(1, int(half_h)))
        cv2.ellipse(vis, (cx_i, cy_i), axes, angle, 0, 360, (255, 0, 255), 3)
        cv2.circle(vis, (cx_i, cy_i), 5, (255, 0, 255), -1)
        lbl = f'{HOOP_DIAMETER_FEET}ft hoop'
        label_r = max(int(half_w), int(half_h))
        cv2.putText(vis, lbl, (cx_i - label_r, cy_i - label_r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, lbl, (cx_i - label_r, cy_i - label_r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Text info panel
    y0 = 30
    dy = 30
    texts = [
        f'Field: {"YES" if found else "NO"}',
        f'Raw line segments: {len(lines)}',
        f'Distinct yard lines (5yd): {len(positions)}',
        f'Dominant angle: {info["dominant_angle_deg"]:.1f} deg' if info['dominant_angle_deg'] is not None else 'Dominant angle: N/A',
        f'Median spacing: {spacing_px:.1f} px' if spacing_px is not None else 'Median spacing: N/A',
        f'Pixels/yard: {info["pixels_per_yard"]:.1f}' if info['pixels_per_yard'] is not None else 'Pixels/yard: N/A',
        f'Hoops detected: {len(hoops)}',
    ]
    if frame_idx is not None:
        texts.insert(0, f'Frame {frame_idx}')

    for i, txt in enumerate(texts):
        cv2.putText(vis, txt, (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(vis, txt, (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis, info


def main():
    parser = argparse.ArgumentParser(description='Visualize field detection and yard lines')
    parser.add_argument('--video', type=str, required=True,
                        help='Video name (same as used with estimate_camera.py)')
    parser.add_argument('--frame', type=int, default=None,
                        help='Specific frame index to visualize (default: middle frame)')
    parser.add_argument('--all_frames', action='store_true',
                        help='Process all frames and save as video')
    parser.add_argument('--sample_every', type=int, default=10,
                        help='When --all_frames, process every Nth frame (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: results/<video>/field_detection.png or .mp4)')
    parser.add_argument('--hoop_mode', action='store_true',
                        help='Enable hoop detection and overlay')
    parser.add_argument('--debug_hoops', action='store_true',
                        help='Save a side-by-side red-mask + contour debug image/video (implies --hoop_mode)')
    args = parser.parse_args()

    video_name = os.path.splitext(args.video)[0]
    img_folder = f'results/{video_name}/images'
    result_dir = f'results/{video_name}'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    if len(imgfiles) == 0:
        if not os.path.exists(args.video):
            print(f'Video file not found: {args.video}')
            sys.exit(1)
        print(f'No extracted frames found — extracting from {args.video} ...')
        os.makedirs(img_folder, exist_ok=True)
        from lib.pipeline import video2frames
        video2frames(args.video, img_folder)
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
        if len(imgfiles) == 0:
            print('Frame extraction failed.')
            sys.exit(1)

    print(f'Found {len(imgfiles)} frames in {img_folder}')

    os.makedirs(result_dir, exist_ok=True)

    if args.all_frames:
        # Process multiple frames, save as video
        output_path = args.output or f'{result_dir}/field_detection.mp4'
        sample_indices = list(range(0, len(imgfiles), args.sample_every))
        print(f'Processing {len(sample_indices)} frames (every {args.sample_every})...')

        writer = None
        debug_writer = None
        success_count = 0
        total_lines = []
        total_hoops = []
        debug_path = f'{result_dir}/field_detection_red_debug.mp4'

        for idx in sample_indices:
            img = cv2.imread(imgfiles[idx])
            if img is None:
                continue
            vis, info = draw_field_detection(img, frame_idx=idx, hoop_mode=args.hoop_mode or args.debug_hoops)

            if writer is None:
                H, W = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, 5.0, (W, H))

            writer.write(vis)

            if args.debug_hoops:
                field_mask_d, _ = detect_field_mask(img)
                dbg = draw_red_debug(img, field_mask_d, frame_idx=idx)
                if debug_writer is None:
                    dH, dW = dbg.shape[:2]
                    debug_writer = cv2.VideoWriter(
                        debug_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (dW, dH))
                debug_writer.write(dbg)
            if info.get('field_found'):
                success_count += 1
                total_lines.append(info.get('num_clusters', 0))
            total_hoops.append(info.get('num_hoops', 0))

        if writer is not None:
            writer.release()
        if debug_writer is not None:
            debug_writer.release()
            print(f'  Red debug video saved to: {debug_path}')

        print(f'\nSummary:')
        print(f'  Frames processed: {len(sample_indices)}')
        print(f'  Field detected: {success_count}/{len(sample_indices)} '
              f'({100*success_count/max(1,len(sample_indices)):.0f}%)')
        if total_lines:
            print(f'  Yard lines per frame: min={min(total_lines)}, '
                  f'max={max(total_lines)}, mean={np.mean(total_lines):.1f}')
        frames_with_hoops = sum(1 for h in total_hoops if h > 0)
        print(f'  Hoops detected: {frames_with_hoops}/{len(sample_indices)} frames '
              f'({100*frames_with_hoops/max(1,len(sample_indices)):.0f}%)')
        print(f'  Saved to: {output_path}')

    else:
        # Single frame mode
        if args.frame is not None:
            idx = args.frame
        else:
            idx = len(imgfiles) // 2  # middle frame

        idx = max(0, min(idx, len(imgfiles) - 1))
        print(f'Processing frame {idx}...')

        img = cv2.imread(imgfiles[idx])
        vis, info = draw_field_detection(img, frame_idx=idx)

        output_path = args.output or f'{result_dir}/field_detection_frame{idx}.png'
        cv2.imwrite(output_path, vis)

        if args.debug_hoops:
            field_mask_d, _ = detect_field_mask(img)
            dbg = draw_red_debug(img, field_mask_d, frame_idx=idx)
            debug_path = f'{result_dir}/field_detection_red_debug_frame{idx}.png'
            cv2.imwrite(debug_path, dbg)
            print(f'Red debug image saved to: {debug_path}')

        print(f'\nResults for frame {idx}:')
        for k, v in info.items():
            print(f'  {k}: {v}')
        print(f'\nSaved to: {output_path}')

        # Also show a multi-frame summary panel (5 evenly spaced frames)
        panel_indices = np.linspace(0, len(imgfiles) - 1, 5, dtype=int)
        panels = []
        for pi in panel_indices:
            img_p = cv2.imread(imgfiles[pi])
            if img_p is None:
                continue
            vis_p, _ = draw_field_detection(img_p, frame_idx=pi, hoop_mode=args.hoop_mode or args.debug_hoops)
            # Resize for panel
            scale = 400 / vis_p.shape[1]
            vis_p = cv2.resize(vis_p, (400, int(vis_p.shape[0] * scale)))
            panels.append(vis_p)

        if panels:
            # Make all same height
            max_h = max(p.shape[0] for p in panels)
            padded = []
            for p in panels:
                if p.shape[0] < max_h:
                    pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                    p = np.vstack([p, pad])
                padded.append(p)
            montage = np.hstack(padded)
            montage_path = f'{result_dir}/field_detection_montage.png'
            cv2.imwrite(montage_path, montage)
            print(f'5-frame montage saved to: {montage_path}')


if __name__ == '__main__':
    main()
