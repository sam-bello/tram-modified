"""
Stage 2 COLMAP reconstruction for camera calibration.

Runs the four steps that turn a folder of frames into a sparse 3-D
reconstruction with self-calibrated intrinsics, using the pycolmap Python API:

  1. feature_extractor  - detects and describes SIFT keypoints
  2. sequential_matcher - matches keypoints between adjacent video frames
  3. incremental_mapping - SfM to recover cameras + sparse point cloud
  4. write_text         - exports binary model to human-readable text

Output layout (siblings of the input colmap_frames/ directory):
    colmap/
        database.db
        sparse/0/        <- binary model (cameras, images, points3D)
        sparse/0_txt/    <- same model in text format for downstream stages

Usage:
    python scripts/run_colmap.py results/myseq/colmap_frames

    # override camera model or SIFT sensitivity
    python scripts/run_colmap.py results/myseq/colmap_frames \\
        --camera-model OPENCV --sift-peak-threshold 0.001

    # resume: skip reconstruction if sparse/0 already exists
    python scripts/run_colmap.py results/myseq/colmap_frames --skip-if-exists
"""

import argparse
import os
import random
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap

# ─── database helpers (sqlite3 — pycolmap.Database is abstract) ──────────────


def _db_images(db_path: Path) -> dict[int, str]:
    """Return {image_id: filename} for all images in the database."""
    con = sqlite3.connect(str(db_path))
    rows = con.execute("SELECT image_id, name FROM images").fetchall()
    con.close()
    return {r[0]: r[1] for r in rows}


def _db_keypoints(db_path: Path, image_id: int) -> np.ndarray:
    """Return (N, 2) float32 array of (x, y) keypoint positions."""
    con = sqlite3.connect(str(db_path))
    row = con.execute(
        "SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)
    ).fetchone()
    con.close()
    if row is None or row[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    nrows, _, blob = row
    kps = np.frombuffer(blob, dtype=np.float32).reshape(nrows, -1)
    return kps[:, :2]  # first two cols are always x, y


def _db_inlier_matches(db_path: Path) -> list[tuple[int, int, np.ndarray]]:
    """
    Return list of (img_id1, img_id2, matches) sorted by descending inlier count.
    matches is a (N, 2) uint32 array of keypoint index pairs.
    """
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0"
    ).fetchall()
    con.close()
    result = []
    for pair_id, nrows, ncols, blob in rows:
        img_id1, img_id2 = pycolmap.pair_id_to_image_pair(pair_id)
        matches = np.frombuffer(blob, dtype=np.uint32).reshape(nrows, 2)
        result.append((img_id1, img_id2, matches))
    result.sort(key=lambda x: -len(x[2]))
    return result


# ─── visualizations ──────────────────────────────────────────────────────────


def visualize_keypoints(db: Path, image_dir: Path, viz_dir: Path,
                        max_frames: int = 30, max_kps: int = 2000):
    """
    Draw SIFT keypoints on each frame and save to viz_dir/keypoints/.

    Keypoints are drawn as small cyan circles.  Frames are subsampled to
    max_frames so the output stays manageable for long sequences.
    max_kps caps the number of points drawn per frame (random subsample).
    """
    out = viz_dir / "keypoints"
    out.mkdir(parents=True, exist_ok=True)

    images = _db_images(db)
    img_ids = sorted(images.keys())

    # Subsample to max_frames, keeping temporal spread
    if len(img_ids) > max_frames:
        idx = np.round(np.linspace(0, len(img_ids) - 1, max_frames)).astype(int)
        img_ids = [img_ids[i] for i in idx]

    print(f"[colmap viz] keypoints: {len(img_ids)} frames -> {out}")
    for img_id in img_ids:
        fname = images[img_id]
        img_path = image_dir / fname
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        kps = _db_keypoints(db, img_id)

        if len(kps) > max_kps:
            idx = np.random.choice(len(kps), max_kps, replace=False)
            kps = kps[idx]

        for x, y in kps.astype(int):
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        n_total = len(_db_keypoints(db, img_id))
        cv2.putText(frame, f"{n_total} keypoints", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        stem = Path(fname).stem
        cv2.imwrite(str(out / f"{stem}_kps.jpg"), frame)

    print(f"[colmap viz] keypoints done  ({len(img_ids)} images)")


def visualize_matches(db: Path, image_dir: Path, viz_dir: Path,
                      max_pairs: int = 9, max_lines: int = 100):
    """
    Draw inlier match lines between the top-N pairs and save to viz_dir/matches/.

    Pairs are ranked by inlier count; up to max_pairs are visualised.
    Images are placed side-by-side with a coloured line per match.
    max_lines caps the number of lines drawn per pair for readability.
    """
    out = viz_dir / "matches"
    out.mkdir(parents=True, exist_ok=True)

    images = _db_images(db)
    pairs = _db_inlier_matches(db)[:max_pairs]

    if not pairs:
        print("[colmap viz] no inlier matches found — skipping match visualisation")
        return

    print(f"[colmap viz] matches: top {len(pairs)} pairs -> {out}")

    # Fixed colour palette — one colour per match line, cycling
    rng = random.Random(0)

    for img_id1, img_id2, matches in pairs:
        fname1 = images.get(img_id1)
        fname2 = images.get(img_id2)
        if not fname1 or not fname2:
            continue
        p1, p2 = image_dir / fname1, image_dir / fname2
        if not p1.exists() or not p2.exists():
            continue

        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Resize to same height for side-by-side layout
        target_h = min(h1, h2, 540)
        s1, s2 = target_h / h1, target_h / h2
        img1 = cv2.resize(img1, (int(w1 * s1), target_h))
        img2 = cv2.resize(img2, (int(w2 * s2), target_h))

        canvas = np.concatenate([img1, img2], axis=1)
        w_left = img1.shape[1]

        kps1 = (_db_keypoints(db, img_id1) * s1).astype(int)
        kps2 = (_db_keypoints(db, img_id2) * s2).astype(int)

        # Subsample lines
        draw_idx = np.arange(len(matches))
        if len(draw_idx) > max_lines:
            draw_idx = np.random.default_rng(0).choice(draw_idx, max_lines, replace=False)

        for i in draw_idx:
            idx1, idx2 = int(matches[i, 0]), int(matches[i, 1])
            if idx1 >= len(kps1) or idx2 >= len(kps2):
                continue
            pt1 = tuple(kps1[idx1])
            pt2 = (kps2[idx2][0] + w_left, kps2[idx2][1])
            color = (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)

        label = (f"{Path(fname1).stem} <-> {Path(fname2).stem} "
                 f"({len(matches)} inliers)")
        cv2.putText(canvas, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        out_name = f"{Path(fname1).stem}_{Path(fname2).stem}_matches.jpg"
        cv2.imwrite(str(out / out_name), canvas)

    print(f"[colmap viz] matches done  ({len(pairs)} pairs)")


# ─── stages ──────────────────────────────────────────────────────────────────


def stage_feature_extraction(db: Path, image_dir: Path,
                              camera_model: str,
                              sift_peak_threshold: float):
    print(f"\n[colmap] 1/4  feature extraction  "
          f"(model={camera_model}, peak_threshold={sift_peak_threshold})")

    reader_opts = pycolmap.ImageReaderOptions(camera_model=camera_model)

    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.sift.peak_threshold = sift_peak_threshold

    pycolmap.extract_features(
        database_path=db,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=extract_opts,
    )
    print(f"[colmap]       done")


def stage_sequential_matching(db: Path, overlap: int):
    print(f"\n[colmap] 2/4  sequential matching  (overlap={overlap})")

    pairing_opts = pycolmap.SequentialPairingOptions()
    pairing_opts.overlap = overlap

    pycolmap.match_sequential(
        database_path=db,
        pairing_options=pairing_opts,
    )
    print(f"[colmap]       done")


def stage_mapper(db: Path, image_dir: Path, sparse_dir: Path) -> dict:
    """Run incremental SfM. Returns the dict of Reconstruction objects."""
    print(f"\n[colmap] 3/4  incremental mapping")
    sparse_dir.mkdir(parents=True, exist_ok=True)

    reconstructions = pycolmap.incremental_mapping(
        database_path=db,
        image_path=image_dir,
        output_path=sparse_dir,
    )
    n = len(reconstructions)
    print(f"[colmap]       done  ({n} reconstruction{'s' if n != 1 else ''} found)")
    return reconstructions


def stage_write_text(reconstruction: pycolmap.Reconstruction, out_dir: Path):
    print(f"\n[colmap] 4/4  writing text model -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write_text(str(out_dir))
    print(f"[colmap]       done")


# ─── quality report ──────────────────────────────────────────────────────────


def _print_quality_report(rec: pycolmap.Reconstruction,
                          n_input_frames: int,
                          warn_low_registration: bool = True):
    n_registered = len(rec.images)
    reg_pct = 100 * n_registered / max(n_input_frames, 1)
    n_points = len(rec.points3D)

    track_lens = [len(pt.track.elements) for pt in rec.points3D.values()]
    mean_track = sum(track_lens) / len(track_lens) if track_lens else 0.0

    print(f"\n[colmap] Reconstruction quality")
    print(f"  Registered frames : {n_registered} / {n_input_frames} ({reg_pct:.0f}%)")
    print(f"  3-D points        : {n_points}")
    print(f"  Mean track length : {mean_track:.1f}")

    # Print camera intrinsics (single-camera model, so camera 1)
    if rec.cameras:
        cam = next(iter(rec.cameras.values()))
        params = [round(p, 2) for p in cam.params]
        print(f"  Camera model      : {cam.model_name}")
        print(f"  Camera params     : {params}")
        w, h = cam.width, cam.height
        if cam.model_name in ("SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"):
            fx = params[0]
            print(f"  Focal length (px) : {fx}  ({fx/max(w,h)*100:.1f}% of max dim)")

    if warn_low_registration and reg_pct < 50:
        print("\n[colmap] WARNING: fewer than 50% of frames registered.")
        print("  Suggested fixes:")
        print("    --sift-peak-threshold 0.001   (find more keypoints on low-texture fields)")
        print("    --camera-model OPENCV         (richer distortion model)")
        print("    Include sideline/stadium texture in the frame crop")

    if n_points < 500:
        print(f"\n[colmap] WARNING: only {n_points} 3-D points — reconstruction may be fragile.")
        print("  Try --sequential-overlap 20 to match more frame pairs.")


# ─── public API ──────────────────────────────────────────────────────────────


def run_colmap_reconstruction(frames_dir: str | os.PathLike,
                              camera_model: str = "SIMPLE_RADIAL",
                              sift_peak_threshold: float = 0.0066,
                              sequential_overlap: int = 10,
                              skip_if_exists: bool = False,
                              visualize: bool = True) -> Path:
    """
    Run the full COLMAP Stage 2 pipeline on a directory of frames.

    Args:
        frames_dir: Directory produced by Stage 1 (extract_colmap_frames.py).
        camera_model: COLMAP camera model.
                      SIMPLE_RADIAL (default) is a good starting point.
                      Try OPENCV if reprojection errors are high.
        sift_peak_threshold: Lower values find more keypoints on low-texture
                             fields (default 0.0066; try 0.001 if <50% of
                             frames register).
        sequential_overlap: Number of adjacent frames each frame is matched
                            against (default 10).
        skip_if_exists: If True and sparse/0 already exists, skip
                        reconstruction and go straight to text export.

    Returns:
        Path to the sparse/0_txt directory.
    """
    frames_dir = Path(frames_dir).resolve()
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"frames_dir does not exist: {frames_dir}")

    seq_dir = frames_dir.parent
    colmap_dir = seq_dir / "colmap"
    db = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_0 = sparse_dir / "0"
    sparse_0_txt = sparse_dir / "0_txt"

    colmap_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    n_frames = len(frame_paths)
    print(f"[colmap] input: {n_frames} frames in {frames_dir}")
    print(f"[colmap] output: {colmap_dir}")

    if skip_if_exists and sparse_0.exists():
        print(f"[colmap] sparse/0 already exists, skipping reconstruction.")
        rec = pycolmap.Reconstruction(str(sparse_0))
        stage_write_text(rec, sparse_0_txt)
        _print_quality_report(rec, n_frames, warn_low_registration=False)
        return sparse_0_txt

    # Remove stale database so feature extraction starts fresh
    if db.exists():
        db.unlink()
        print(f"[colmap] removed stale database")

    viz_dir = colmap_dir / "viz"

    stage_feature_extraction(db, frames_dir, camera_model, sift_peak_threshold)
    if visualize:
        visualize_keypoints(db, frames_dir, viz_dir)

    stage_sequential_matching(db, sequential_overlap)
    if visualize:
        visualize_matches(db, frames_dir, viz_dir)

    reconstructions = stage_mapper(db, frames_dir, sparse_dir)

    if not reconstructions:
        print("\n[colmap] ERROR: mapper produced no reconstruction.", file=sys.stderr)
        print("  Try: --sift-peak-threshold 0.001  or  --camera-model OPENCV",
              file=sys.stderr)
        sys.exit(1)

    # COLMAP may produce multiple sub-reconstructions; take the largest
    rec = max(reconstructions.values(), key=lambda r: len(r.images))
    stage_write_text(rec, sparse_0_txt)
    _print_quality_report(rec, n_frames)

    print(f"\n[colmap] Done.  Text model -> {sparse_0_txt}")
    print(f"  Next step (Stage 3): annotate field landmarks in reference frames,")
    print(f"  then run scripts/align_reconstruction.py {sparse_0_txt}")
    return sparse_0_txt


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="Stage 2: COLMAP sparse reconstruction for camera calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("frames_dir",
                   help="Frame directory from Stage 1 (e.g. results/myseq/colmap_frames)")
    p.add_argument("--camera-model", default="SIMPLE_RADIAL",
                   choices=["SIMPLE_RADIAL", "OPENCV", "FULL_OPENCV",
                            "SIMPLE_PINHOLE", "PINHOLE", "RADIAL"],
                   help="COLMAP camera model")
    p.add_argument("--sift-peak-threshold", type=float, default=0.0066,
                   help="SIFT peak threshold; lower = more keypoints "
                        "(try 0.001 on low-texture footage)")
    p.add_argument("--sequential-overlap", type=int, default=10,
                   help="Number of adjacent frames to match per frame")
    p.add_argument("--skip-if-exists", action="store_true",
                   help="Skip reconstruction if sparse/0 already exists")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip keypoint and match visualizations")
    args = p.parse_args()

    run_colmap_reconstruction(
        frames_dir=args.frames_dir,
        camera_model=args.camera_model,
        sift_peak_threshold=args.sift_peak_threshold,
        sequential_overlap=args.sequential_overlap,
        skip_if_exists=args.skip_if_exists,
        visualize=not args.no_viz,
    )


if __name__ == "__main__":
    main()
