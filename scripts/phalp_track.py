"""
PHALP tracking — replaces DEVA as the first step of the TRAM pipeline.

Run this script with the 4dhumans conda environment BEFORE estimate_camera.py.
It extracts video frames (if not already done), runs PHALP tracking, and saves:
  - results/{seq}/tracks_phalp.pkl   -- PHALP tracks for estimate_humans.py
  - results/{seq}/masks_phalp.npy    -- per-frame bbox masks for DROID-SLAM

estimate_camera.py auto-detects masks_phalp.npy and skips DEVA when it is present.

Recommended pipeline:
    conda run -n 4dhumans python scripts/phalp_track.py --video path/to/video.mp4
    conda run -n tram     python scripts/estimate_camera.py --video path/to/video.mp4
    conda run -n tram     python scripts/estimate_humans.py --video path/to/video.mp4

Why PHALP instead of DEVA:
  - PHALP uses 3D-pose similarity (HMAR embeddings) for re-identification, so it
    bridges the gap when DEVA would create a new track ID for the same person.
  - PHALP's DeepSort keeps a confirmed track alive for up to max_age_track frames
    even without a fresh detection, providing a valid bbox for gap frames.
  - All confirmed PHALP frames are marked det=True so HMR_VIMO sees a single
    unbroken (or minimally-broken) sequence instead of many short chunks.
  - Removes the ViTDet + SAM + DEVA per-frame inference from estimate_camera.py,
    cutting pipeline time roughly in half and eliminating the DEVA-related crashes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import pickle
import numpy as np
import torch
import cv2
import joblib
from glob import glob

from omegaconf import OmegaConf
from phalp.configs.base import FullConfig
from phalp.trackers.PHALP import PHALP


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path, img_folder):
    """Extract video frames to img_folder as 0000.jpg, 0001.jpg, ..."""
    os.makedirs(img_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f'{img_folder}/{count:04d}.jpg', frame)
        count += 1
    cap.release()
    return count


def save_slam_masks(tracks, img_folder, out_path):
    """Save per-frame bounding-box masks derived from PHALP tracks.

    These rectangular masks tell DROID-SLAM which pixels are moving humans so
    it can ignore them during camera estimation.  Saved as a uint8 numpy array
    of shape (N, H, W) where 1 = human region.
    """
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if not imgfiles:
        print('  [masks] no images found, skipping mask save')
        return

    img0 = cv2.imread(imgfiles[0])
    H, W = img0.shape[:2]
    n = len(imgfiles)

    masks = np.zeros((n, H, W), dtype=np.uint8)
    for entries in tracks.values():
        for entry in entries:
            f = entry['frame']
            if f >= n:
                continue
            # det_box is stored as [[x1, y1, x2, y2, conf]] after serialisation
            box = entry['det_box'][0]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                masks[f, y1:y2, x1:x2] = 1

    np.save(out_path, masks)
    print(f'  Saved SLAM masks {masks.shape} → {out_path}')

def phalp_bbox_xywh_to_xyxy(bbox):
    """PHALP stores bbox as [x, y, w, h] — convert to [x1, y1, x2, y2]."""
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def convert_phalp_to_tram_tracks(final_visuals_dic):
    """Convert PHALP's final_visuals_dic to TRAM's tracks dict.

    TRAM tracks format (one entry per frame per track):
        {track_id: [{'frame': int, 'det': bool,
                     'det_box': np.ndarray (1,5),   # [x1,y1,x2,y2,conf]
                     'seg_box': np.ndarray (1,4),   # [x1,y1,x2,y2]
                     'id': int}, ...]}

    For frames where PHALP has an active (confirmed) track, we set det=True
    regardless of tracked_time so that HMR_VIMO's valid-filter does not drop
    gap frames.  PHALP provides a valid bbox even for predicted frames (last
    known position from the DeepSort state), which is far better than the zero
    box DEVA would leave behind.
    """
    tracks = {}

    for frame_name in sorted(final_visuals_dic.keys()):
        # Derive integer frame index from filename, e.g. "0042.jpg" -> 42
        basename = os.path.splitext(os.path.basename(frame_name))[0]
        try:
            frame_idx = int(basename)
        except ValueError:
            # If filename is not purely numeric, fall back to insertion order
            frame_idx = final_visuals_dic[frame_name].get('time', 0)

        frame_data = final_visuals_dic[frame_name]
        tids          = frame_data.get('tid',          [])
        bboxes        = frame_data.get('bbox',         [])
        confs         = frame_data.get('conf',         [])
        tracked_times = frame_data.get('tracked_time', [])

        for i, tid in enumerate(tids):
            if i >= len(bboxes):
                continue

            bbox_xywh = np.asarray(bboxes[i], dtype=np.float32)
            bbox_xyxy = phalp_bbox_xywh_to_xyxy(bbox_xywh)

            # Use actual detection confidence when available; fall back to 0.
            conf = float(confs[i]) if i < len(confs) else 0.0
            tracked_time = int(tracked_times[i]) if i < len(tracked_times) else 0

            # Mark det=True for all confirmed PHALP frames.
            # tracked_time==0  → freshly detected this frame  (high confidence)
            # tracked_time>0   → Kalman-predicted / last-known position
            # Either way PHALP provides a valid bbox, so we keep it in the
            # VIMO inference path.  Confidence is set to 0 for predicted frames
            # so downstream code can tell them apart if needed.
            if tracked_time > 0:
                conf = 0.0   # flag as non-fresh

            det_box = np.array([[*bbox_xyxy, conf]], dtype=np.float32)  # (1, 5)
            seg_box = bbox_xyxy.reshape(1, 4)                           # (1, 4)

            entry = {
                'frame':        frame_idx,
                'det':          True,       # always True — PHALP bbox is valid
                'det_box':      det_box,
                'seg_box':      seg_box,
                'id':           tid,
                'tracked_time': tracked_time,
            }

            tracks.setdefault(tid, []).append(entry)

    # Sort each track by ascending frame index
    for tid in tracks:
        tracks[tid].sort(key=lambda e: e['frame'])

    # Convert ALL numpy scalars/arrays to plain Python types so the file can
    # be loaded by the tram conda env, which has numpy 1.x (this env uses
    # numpy 2.x).  numpy 2.x pickles arrays with 'numpy._core' class
    # references that numpy 1.x cannot resolve.  np.concatenate in
    # estimate_humans.py handles nested lists just fine.
    for tid in list(tracks.keys()):
        # Re-key with a plain Python int in case tid is a numpy integer
        py_tid = int(tid)
        entries = tracks.pop(tid)
        for entry in entries:
            if isinstance(entry.get('det_box'), np.ndarray):
                entry['det_box'] = entry['det_box'].tolist()
            if isinstance(entry.get('seg_box'), np.ndarray):
                entry['seg_box'] = entry['seg_box'].tolist()
            entry['frame']        = int(entry['frame'])
            entry['id']           = int(entry['id']) if entry['id'] is not None else None
            entry['tracked_time'] = int(entry['tracked_time'])
            entry['det']          = bool(entry['det'])
        tracks[py_tid] = entries

    return tracks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True,
                        help='Input video path (same as estimate_camera.py)')
    parser.add_argument('--max_age', type=int, default=50,
                        help='Max frames a PHALP track can coast without a '
                             'detection before it is deleted (default 50)')
    args = parser.parse_args()

    file       = args.video
    seq        = os.path.basename(file).split('.')[0]
    seq_folder = f'results/{seq}'
    img_folder = f'{seq_folder}/images'
    phalp_out  = f'{seq_folder}/phalp_out'

    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(phalp_out, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract frames if not already done
    # ------------------------------------------------------------------
    existing = sorted(glob(f'{img_folder}/*.jpg'))
    if not existing:
        print('Extracting frames...')
        n = extract_frames(file, img_folder)
        print(f'  {n} frames extracted to {img_folder}')
    else:
        print(f'  frames already extracted ({len(existing)} images), skipping')

    # ------------------------------------------------------------------
    # Build PHALP config
    # ------------------------------------------------------------------
    cfg = OmegaConf.structured(FullConfig)

    # IO: read from the already-extracted frame folder
    cfg.video.source     = img_folder
    cfg.video.output_dir = phalp_out

    # Tracking parameters
    cfg.phalp.max_age_track = args.max_age
    cfg.phalp.n_init        = 3       # frames before a track is confirmed
    cfg.phalp.low_th_c      = 0.5    # detection confidence threshold

    # Skip rendering (we only need the tracking output)
    cfg.render.enable = False

    # Always recompute (don't skip if a previous pkl exists)
    cfg.overwrite = True

    cfg.detect_shots = False

    # ------------------------------------------------------------------
    # Run PHALP
    # ------------------------------------------------------------------
    print('Initialising PHALP tracker...')
    tracker = PHALP(cfg)

    print('Tracking...')
    with torch.no_grad():
        result = tracker.track()

    # track() returns 0 when it reuses a cached pkl
    if result == 0:
        pkl_files = sorted(glob(f'{phalp_out}/results/*.pkl'))
        if not pkl_files:
            raise RuntimeError('No PHALP result pkl found in ' + phalp_out)
        print(f'Loading cached result from {pkl_files[-1]}')
        final_visuals_dic = joblib.load(pkl_files[-1])
    else:
        final_visuals_dic, pkl_path = result
        print(f'PHALP result saved to {pkl_path}')

    # ------------------------------------------------------------------
    # Convert to TRAM format and overwrite tracks.npy
    # ------------------------------------------------------------------
    print('Converting PHALP tracks to TRAM format...')
    tracks = convert_phalp_to_tram_tracks(final_visuals_dic)

    n_tracks = len(tracks)
    total_frames = sum(len(v) for v in tracks.values())
    print(f'  {n_tracks} tracks,  {total_frames} total track-frame entries')

    # Save as a plain-Python pickle (protocol 2) so numpy 1.x in the tram
    # env can load it without 'numpy._core' errors.  estimate_humans.py
    # will prefer this file over the DEVA tracks.npy when it exists.
    out_path = f'{seq_folder}/tracks_phalp.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(tracks, f, protocol=2)
    print(f'Saved tracks to {out_path}')

    # ------------------------------------------------------------------
    # Save per-frame bbox masks for DROID-SLAM
    # estimate_camera.py will load these and skip DEVA automatically.
    # ------------------------------------------------------------------
    masks_path = f'{seq_folder}/masks_phalp.npy'
    print('Saving SLAM masks from PHALP bounding boxes...')
    save_slam_masks(tracks, img_folder, masks_path)


if __name__ == '__main__':
    main()
