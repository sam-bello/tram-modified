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
    compute_yard_line_spacing_pixels, estimate_field_scale,
    YARD_IN_METERS, YARD_LINE_SPACING_YARDS
)


def draw_field_detection(image, frame_idx=None):
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

    # 3) Cluster into distinct yard lines
    positions, perp_dir = cluster_yard_lines(lines, dominant_angle) if dominant_angle is not None else (np.array([]), None)
    info['num_clusters'] = len(positions)

    # 4) Compute spacing
    spacing_px = compute_yard_line_spacing_pixels(lines, dominant_angle) if dominant_angle is not None and len(lines) >= 2 else None
    info['spacing_px'] = spacing_px
    if spacing_px is not None:
        info['pixels_per_yard'] = spacing_px / YARD_LINE_SPACING_YARDS
    else:
        info['pixels_per_yard'] = None

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

    # Text info panel
    y0 = 30
    dy = 30
    texts = [
        f'Field: {"YES" if found else "NO"}',
        f'Raw line segments: {len(lines)}',
        f'Distinct yard lines: {len(positions)}',
        f'Dominant angle: {info["dominant_angle_deg"]:.1f} deg' if info['dominant_angle_deg'] is not None else 'Dominant angle: N/A',
        f'Median spacing: {spacing_px:.1f} px' if spacing_px is not None else 'Median spacing: N/A',
        f'Pixels/yard: {info["pixels_per_yard"]:.1f}' if info['pixels_per_yard'] is not None else 'Pixels/yard: N/A',
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
    args = parser.parse_args()

    video_name = os.path.splitext(args.video)[0]
    img_folder = f'results/{video_name}/images'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    if len(imgfiles) == 0:
        print(f'No images found in {img_folder}')
        print('Run estimate_camera.py first to extract frames.')
        sys.exit(1)

    print(f'Found {len(imgfiles)} frames in {img_folder}')

    result_dir = f'results/{video_name}'
    os.makedirs(result_dir, exist_ok=True)

    if args.all_frames:
        # Process multiple frames, save as video
        output_path = args.output or f'{result_dir}/field_detection.mp4'
        sample_indices = list(range(0, len(imgfiles), args.sample_every))
        print(f'Processing {len(sample_indices)} frames (every {args.sample_every})...')

        writer = None
        success_count = 0
        total_lines = []

        for idx in sample_indices:
            img = cv2.imread(imgfiles[idx])
            if img is None:
                continue
            vis, info = draw_field_detection(img, frame_idx=idx)

            if writer is None:
                H, W = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, 5.0, (W, H))

            writer.write(vis)
            if info.get('field_found'):
                success_count += 1
                total_lines.append(info.get('num_clusters', 0))

        if writer is not None:
            writer.release()

        print(f'\nSummary:')
        print(f'  Frames processed: {len(sample_indices)}')
        print(f'  Field detected: {success_count}/{len(sample_indices)} '
              f'({100*success_count/max(1,len(sample_indices)):.0f}%)')
        if total_lines:
            print(f'  Yard lines per frame: min={min(total_lines)}, '
                  f'max={max(total_lines)}, mean={np.mean(total_lines):.1f}')
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
            vis_p, _ = draw_field_detection(img_p, frame_idx=pi)
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
