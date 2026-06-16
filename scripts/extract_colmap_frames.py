"""
Stage 1 frame extraction for COLMAP-based camera calibration.

Analyses camera motion across the video and samples sparsely from static
segments (where the camera barely moves) and densely from moving segments
(where parallax enables SfM self-calibration).  Targets 30-100 frames total
so COLMAP runs quickly and registers as many views as possible.

Usage:
    python scripts/extract_colmap_frames.py <video>
    python scripts/extract_colmap_frames.py <video> --output results/myseq/colmap_frames
    python scripts/extract_colmap_frames.py <video> --moving-fps 10 --max-frames 120
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + "/..")

# ─── motion analysis ────────────────────────────────────────────────────────


def _estimate_camera_motion_scores(cap: cv2.VideoCapture, sample_every: int = 3) -> tuple:
    """
    Estimate per-frame camera translation magnitude using sparse optical flow.

    Shi-Tomasi corners are tracked frame-to-frame with Lucas-Kanade.  A
    homography is fit with RANSAC so that localised player motion does not
    inflate the score; the translation component of that homography gives the
    camera-motion score.

    Args:
        cap: Opened cv2.VideoCapture, rewound to frame 0.
        sample_every: Only compute flow every N frames to keep it fast;
                      scores for skipped frames are linearly interpolated.

    Returns:
        (motion_scores, fps, n_frames) where motion_scores is float32 (n_frames,).
    """
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Down-sample to this width for speed
    PROC_W = 480

    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=7)

    sampled_indices: list[int] = []
    sampled_scores: list[float] = []

    prev_gray: np.ndarray | None = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if fi % sample_every != 0:
            continue

        h, w = frame.shape[:2]
        scale = PROC_W / w
        small = cv2.resize(frame, (PROC_W, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        score = 0.0
        if prev_gray is not None:
            pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
            if pts is not None and len(pts) >= 8:
                pts1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk_params)
                good_src = pts[status.ravel() == 1]
                good_dst = pts1[status.ravel() == 1]
                if len(good_src) >= 8:
                    H, mask = cv2.findHomography(good_src, good_dst, cv2.RANSAC, 3.0)
                    if H is not None:
                        # Translation component of the homography (in down-sampled px)
                        tx, ty = H[0, 2], H[1, 2]
                        score = float(np.sqrt(tx ** 2 + ty ** 2))

        sampled_indices.append(fi)
        sampled_scores.append(score)
        prev_gray = gray

    # Interpolate back to every frame
    full_idx = np.arange(n_frames)
    motion_scores = np.interp(full_idx,
                              np.array(sampled_indices, dtype=float),
                              np.array(sampled_scores, dtype=float)).astype(np.float32)
    return motion_scores, fps, n_frames


def _classify_motion(motion_scores: np.ndarray,
                     threshold: float | None = None,
                     smooth_window: int = 15) -> tuple:
    """
    Smooth scores and return a bool mask (True = camera is moving).

    The threshold defaults to the 35th percentile of non-zero scores, which
    typically separates static-camera noise from genuine translation.
    """
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smoothed = np.convolve(motion_scores, kernel, mode="same")

    nonzero = smoothed[smoothed > 0]
    if threshold is None:
        threshold = float(np.percentile(nonzero, 35)) if len(nonzero) else 1.0

    is_moving = smoothed > threshold
    return is_moving, smoothed, threshold


# ─── frame selection ─────────────────────────────────────────────────────────


def _select_frame_indices(n_frames: int,
                          is_moving: np.ndarray,
                          fps: float,
                          static_fps: float = 1.0,
                          moving_fps: float = 5.0,
                          target_min: int = 30,
                          target_max: int = 100) -> list[int]:
    """
    Choose which frame indices to extract.

    Static segments are sampled at `static_fps`; moving segments at
    `moving_fps`.  If the resulting count falls outside [target_min,
    target_max] the intervals are rescaled uniformly to compensate.
    """
    static_step = max(1, round(fps / static_fps))
    moving_step = max(1, round(fps / moving_fps))

    selected: set[int] = set()
    for i in range(n_frames):
        step = moving_step if is_moving[i] else static_step
        if i % step == 0:
            selected.add(i)

    # Always include first and last frame
    selected.add(0)
    selected.add(n_frames - 1)
    selected_list = sorted(selected)

    # Pad up if too few
    if len(selected_list) < target_min:
        step = max(1, n_frames // target_min)
        extra = set(range(0, n_frames, step))
        selected_list = sorted(set(selected_list) | extra)

    # Thin down if too many (uniform sub-sample to preserve temporal spread)
    if len(selected_list) > target_max:
        keep_idx = np.round(np.linspace(0, len(selected_list) - 1, target_max)).astype(int)
        selected_list = [selected_list[k] for k in keep_idx]

    return selected_list


# ─── public API ──────────────────────────────────────────────────────────────


def extract_colmap_frames(video_path: str | os.PathLike,
                          output_dir: str | os.PathLike,
                          static_fps: float = 1.0,
                          moving_fps: float = 5.0,
                          target_min: int = 30,
                          target_max: int = 100,
                          motion_threshold: float | None = None,
                          jpeg_quality: int = 95) -> list[str]:
    """
    Extract frames suitable for COLMAP reconstruction.

    Analyses per-frame camera motion with sparse optical flow, classifies each
    frame as static or moving, then samples at different rates so COLMAP
    receives enough parallax to self-calibrate while the total count stays
    within [target_min, target_max].

    Args:
        video_path: Path to input video.
        output_dir: Directory to write JPEG frames into (created if absent).
        static_fps: Frames-per-second to sample during static camera segments.
        moving_fps: Frames-per-second to sample during moving camera segments.
        target_min: Floor on total frames extracted.
        target_max: Ceiling on total frames extracted.
        motion_threshold: Manual threshold on the camera-motion score;
                          auto-detected when None.
        jpeg_quality: JPEG compression quality for saved frames (0-100).

    Returns:
        Sorted list of absolute paths to the extracted JPEG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    n_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_raw = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames_raw / fps_raw
    print(f"[extract_colmap_frames] {Path(video_path).name}: "
          f"{w}x{h}  {fps_raw:.2f} fps  {n_frames_raw} frames  ({duration:.1f}s)")

    # Sample every 3rd frame during analysis (balances speed vs. resolution)
    sample_every = max(1, round(fps_raw / 10))
    print(f"[extract_colmap_frames] analysing motion "
          f"(sampling every {sample_every} frames for speed)...")
    motion_scores, fps, n_frames = _estimate_camera_motion_scores(cap, sample_every)

    is_moving, smoothed, threshold = _classify_motion(motion_scores, motion_threshold)
    n_moving = int(is_moving.sum())
    n_static = n_frames - n_moving
    print(f"[extract_colmap_frames] motion threshold={threshold:.2f}  "
          f"static={n_static} frames ({100*n_static/n_frames:.0f}%)  "
          f"moving={n_moving} frames ({100*n_moving/n_frames:.0f}%)")

    frame_indices = _select_frame_indices(
        n_frames, is_moving, fps,
        static_fps=static_fps, moving_fps=moving_fps,
        target_min=target_min, target_max=target_max)
    print(f"[extract_colmap_frames] extracting {len(frame_indices)} frames...")

    # Sequential read (faster than random seek for many frames)
    idx_set = set(frame_indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    extracted: list[str] = []
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fi in idx_set:
            out_path = output_dir / f"frame_{fi:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            extracted.append(str(out_path.resolve()))
        fi += 1
        if fi > max(frame_indices):
            break
    cap.release()

    # Save motion analysis so the user can inspect and re-run with --threshold
    np.savez(str(output_dir / "motion_analysis.npz"),
             motion_scores=motion_scores,
             smoothed_scores=smoothed,
             is_moving=is_moving,
             threshold=np.float32(threshold),
             selected_frames=np.array(frame_indices, dtype=np.int32),
             fps=np.float32(fps))

    print(f"[extract_colmap_frames] saved {len(extracted)} frames -> {output_dir}")
    print(f"[extract_colmap_frames] motion analysis -> {output_dir / 'motion_analysis.npz'}")
    return extracted


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1: extract frames from a football field video for COLMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("video", help="Input video file")
    p.add_argument("--output", "-o", default=None,
                   help="Output directory; defaults to results/<seq>/colmap_frames")
    p.add_argument("--static-fps", type=float, default=1.0,
                   help="Sampling rate (fps) for static camera segments")
    p.add_argument("--moving-fps", type=float, default=5.0,
                   help="Sampling rate (fps) for moving camera segments")
    p.add_argument("--min-frames", type=int, default=30,
                   help="Minimum total frames to extract")
    p.add_argument("--max-frames", type=int, default=100,
                   help="Maximum total frames to extract")
    p.add_argument("--threshold", type=float, default=None,
                   help="Manual motion-score threshold; auto-detected when omitted")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality for saved frames (0-100)")
    return p


def main():
    args = _build_parser().parse_args()

    video_path = args.video
    if args.output is None:
        seq = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join("results", seq, "colmap_frames")
    else:
        output_dir = args.output

    paths = extract_colmap_frames(
        video_path=video_path,
        output_dir=output_dir,
        static_fps=args.static_fps,
        moving_fps=args.moving_fps,
        target_min=args.min_frames,
        target_max=args.max_frames,
        motion_threshold=args.threshold,
        jpeg_quality=args.jpeg_quality,
    )

    print(f"\nDone. {len(paths)} frames in {output_dir}")
    print("Next step (Stage 2):  colmap feature_extractor "
          f"--database_path {os.path.join(os.path.dirname(output_dir), 'database.db')} "
          f"--image_path {output_dir} "
          "--ImageReader.single_camera 1 "
          "--ImageReader.camera_model SIMPLE_RADIAL")


if __name__ == "__main__":
    main()
