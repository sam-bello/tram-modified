"""
Export SMPL 3D keypoints to JSON for every video in a folder.

For each video the script automatically runs the full TRAM pipeline if the
results are not already present:
    Step 1  phalp_track.py        (4dhumans env) -- replace DEVA with PHALP 
    Step 2  estimate_camera.py    (tram env)   -- SLAM tracking
    Step 3  estimate_humans.py    (tram env)   -- HMR-VIMO pose estimation
    Step 4  export                              -- write keypoints JSON

Previously-completed steps are detected by their output files and skipped,
so re-running the script on a partially-processed folder is safe.

Usage (single video):
    conda run -n tram python scripts/export_keypoints.py --video Ravens_trimmed/2022_BARNO_AMARE_DL25.mp4 --field_mode

Usage (all videos in a folder):
    conda run -n tram python scripts/export_keypoints.py --video_dir Ravens_trimmed/ --out_dir keypoints/ --field_mode

Optional flags passed through to estimate_camera.py:
    --field_mode        use yard-line scale + field-plane alignment
    --static_camera     treat the camera as static

Output format matches 2022_BARNO_AMARE_DL25.json:
{
  "player_id": "2022_BARNO_AMARE_DL25",
  "year": 2022,
  "n_keypoints": 45,
  "athlete_frames": [
    {"frame": 0, "keypoints": [[x, y, z], ...]},   // 45 world-space 3D joints
    ...
  ]
}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import json
import argparse
import subprocess
import time
import numpy as np
import torch
from glob import glob

from lib.models.smpl import SMPL

# Absolute path to the tram repo root (scripts live one level below it)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Python for the tram env (this process)
PYTHON_TRAM = sys.executable

# Python for the 4dhumans env (needed for phalp_track.py)
_4D_PYTHON_CANDIDATES = [
    os.path.expanduser('~/.conda/envs/4dhumans/python.exe'),
    r'C:\Users\007sb\.conda\envs\4dhumans\python.exe',
    '/home/user/.conda/envs/4dhumans/bin/python',  # Linux fallback
]
PYTHON_4D = next((p for p in _4D_PYTHON_CANDIDATES if os.path.exists(p)), None)


# ---------------------------------------------------------------------------
# pipeline helpers
# ---------------------------------------------------------------------------

def _flush_gpu(wait_s=5):
    """Nudge the Windows GPU driver to reclaim VRAM from the previous process.

    Strategy:
      1. Poll nvidia-smi until memory stops dropping (or timeout).
      2. Spin up a tiny CUDA process that initialises the runtime, calls
         empty_cache()+synchronize(), then exits — this forces the driver to
         process any pending context-destruction from the previous step.
      3. Sleep briefly so the driver has time to finish the cleanup.
    """
    # --- 1. Wait for nvidia-smi memory to stabilise ---
    try:
        prev_used = None
        deadline = time.time() + 20          # max 20 s of polling
        while time.time() < deadline:
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            used = int(r.stdout.strip().split('\n')[0])
            if prev_used is not None and used >= prev_used:
                break                        # memory has stopped falling
            prev_used = used
            time.sleep(1)
    except Exception:
        pass

    # --- 2. Tiny CUDA context to force driver GC ---
    try:
        subprocess.run(
            [PYTHON_TRAM, '-c',
             'import torch; torch.cuda.init(); '
             'torch.cuda.empty_cache(); torch.cuda.synchronize()'],
            capture_output=True, timeout=30, cwd=ROOT_DIR)
    except Exception:
        pass

    # --- 3. Short grace period ---
    time.sleep(wait_s)


def _run(cmd, step_name):
    """Run a subprocess from ROOT_DIR. Returns True on success."""
    print(f'    [{step_name}] {" ".join(os.path.basename(c) for c in cmd[:3])}...')
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    if result.returncode != 0:
        print(f'    [{step_name}] FAILED (exit {result.returncode})')
        return False
    return True


def run_pipeline(video_path, seq_folder, field_mode=False, static_camera=False,
                 hoop_mode=False):
    """Run whichever pipeline steps have not yet produced their output files.

    Step order:
      1. phalp_track.py   (4dhumans env) — extracts frames + PHALP tracking +
                                           saves masks_phalp.npy for SLAM
      2. estimate_camera.py (tram env)   — DROID-SLAM using PHALP masks (DEVA skipped)
      3. estimate_humans.py (tram env)   — HMR-VIMO pose estimation

    Returns True if all required outputs exist after running.
    """
    video_abs = os.path.abspath(video_path)

    # ------------------------------------------------------------------
    # Step 1: phalp_track.py  →  tracks_phalp.pkl + masks_phalp.npy
    #         also extracts video frames into results/{seq}/images/
    # ------------------------------------------------------------------
    if not os.path.exists(f'{seq_folder}/tracks_phalp.pkl'):
        if PYTHON_4D is None:
            print('    [phalp_track] ERROR: 4dhumans Python not found. '
                  'Set PYTHON_4D in the script or install the env.')
            return False
        cmd = [PYTHON_4D, 'scripts/phalp_track.py', '--video', video_abs]
        if not _run(cmd, 'phalp_track'):
            return False
        _flush_gpu()
    else:
        print('    [phalp_track] already done, skipping')

    # ------------------------------------------------------------------
    # Step 2: estimate_camera.py  →  camera.npy
    #         auto-detects masks_phalp.npy and skips DEVA
    # ------------------------------------------------------------------
    if not os.path.exists(f'{seq_folder}/camera.npy'):
        cmd = [PYTHON_TRAM, 'scripts/estimate_camera.py', '--video', video_abs]
        if field_mode:
            cmd.append('--field_mode')
        if hoop_mode:
            cmd.append('--hoop_mode')
        if static_camera:
            cmd.append('--static_camera')
        if not _run(cmd, 'estimate_camera'):
            return False
        _flush_gpu()
    else:
        print('    [estimate_camera] already done, skipping')

    # ------------------------------------------------------------------
    # Step 3: estimate_humans.py  →  hps/hps_track_0.npy
    # ------------------------------------------------------------------
    if not os.path.exists(f'{seq_folder}/hps/hps_track_0.npy'):
        cmd = [PYTHON_TRAM, 'scripts/estimate_humans.py', '--video', video_abs]
        if not _run(cmd, 'estimate_humans'):
            return False
        _flush_gpu()
    else:
        print('    [estimate_humans] already done, skipping')

    return True


# ---------------------------------------------------------------------------
# keypoint export
# ---------------------------------------------------------------------------

def export_keypoints(seq_folder, smpl_model, device='cuda'):
    """Compute world-space 3D keypoints for the primary human track.

    Returns (athlete_frames, n_keypoints) or None on error.
    """
    hps_path = f'{seq_folder}/hps/hps_track_0.npy'
    cam_path = f'{seq_folder}/camera.npy'

    hps = np.load(hps_path, allow_pickle=True).item()
    cam = np.load(cam_path, allow_pickle=True).item()

    pred_rotmat = hps['pred_rotmat'].to(device)   # (N, 24, 3, 3)
    pred_shape  = hps['pred_shape'].to(device)    # (N, 10)
    pred_trans  = hps['pred_trans'].to(device)    # (N, 1, 3)
    frame       = hps['frame']                    # (N,) int tensor

    world_cam_R = torch.tensor(cam['world_cam_R'], dtype=torch.float32).to(device)
    world_cam_T = torch.tensor(cam['world_cam_T'], dtype=torch.float32).to(device)

    # Average shape across the track for temporal stability (matches visualizer)
    mean_shape = pred_shape.mean(dim=0, keepdim=True).expand_as(pred_shape)

    with torch.no_grad():
        # default_smpl=True → raw smplx 45-joint output (no JOINT_MAP remapping)
        pred = smpl_model(
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=mean_shape,
            transl=pred_trans.squeeze(1),
            pose2rot=False,
            default_smpl=True,
        )

    # Transform from SMPL camera space → scene world space
    cam_r = world_cam_R[frame]   # (N, 3, 3)
    cam_t = world_cam_T[frame]   # (N, 3)
    joints_world = torch.einsum('bij,bnj->bni', cam_r, pred.joints) + cam_t[:, None]
    joints_world = joints_world.cpu().numpy()    # (N, 45, 3)

    athlete_frames = []
    for i, f in enumerate(frame.tolist()):
        kp = [[round(v, 4) for v in pt] for pt in joints_world[i].tolist()]
        athlete_frames.append({'frame': int(f), 'keypoints': kp})

    athlete_frames.sort(key=lambda e: e['frame'])
    return athlete_frames, int(joints_world.shape[1])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str,
                       help='Single input video path')
    group.add_argument('--video_dir', type=str,
                       help='Folder of .mp4 files to process in batch')
    parser.add_argument('--out_dir', type=str, default='keypoints',
                        help='Output directory for JSON files (default: keypoints/)')
    parser.add_argument('--field_mode', action='store_true',
                        help='Pass --field_mode to estimate_camera.py')
    parser.add_argument('--hoop_mode', action='store_true',
                        help='Pass --hoop_mode to estimate_camera.py (requires --field_mode)')
    parser.add_argument('--static_camera', action='store_true',
                        help='Pass --static_camera to estimate_camera.py')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ref_dir', type=str, default='pose_json_3d_45_4dhumans',
                        help='Directory of reference JSON files whose starting frame '
                             'is used to offset output frame numbers (default: '
                             'pose_json_3d_45_4dhumans/)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    videos = [args.video] if args.video else sorted(glob(os.path.join(args.video_dir, '*.mp4')))
    if not videos:
        print(f'No .mp4 files found.')
        return

    device = args.device if torch.cuda.is_available() else 'cpu'

    n_ok = n_fail = 0
    for video_path in videos:
        seq = os.path.basename(video_path).rsplit('.', 1)[0]
        seq_folder = os.path.join(ROOT_DIR, 'results', seq)
        out_path = os.path.join(args.out_dir, f'{seq}.json')

        print(f'\n[{seq}]')

        # Run pipeline steps that haven't been done yet.
        # SMPL is NOT loaded yet so the subprocesses (DROID-SLAM, PHALP,
        # HMR-VIMO) get the full GPU budget.
        ok = run_pipeline(
            video_path, seq_folder,
            field_mode=args.field_mode,
            hoop_mode=args.hoop_mode,
            static_camera=args.static_camera,
        )
        if not ok:
            print(f'  Pipeline failed, skipping export.')
            n_fail += 1
            continue

        # Load SMPL only after the heavy subprocesses have finished and
        # released their GPU memory.
        print(f'  Loading SMPL model...')
        smpl = SMPL().to(device).eval()

        # Export keypoints
        try:
            athlete_frames, n_kp = export_keypoints(seq_folder, smpl, device=device)
        except Exception as e:
            print(f'  Export failed: {e}')
            n_fail += 1
            del smpl
            torch.cuda.empty_cache()
            continue

        # Free SMPL before the next video's pipeline runs
        del smpl
        torch.cuda.empty_cache()

        # Align starting frame to the reference file when one exists
        ref_path = os.path.join(ROOT_DIR, args.ref_dir, f'{seq}.json')
        frame_offset = 0
        if os.path.exists(ref_path):
            with open(ref_path) as fref:
                ref_doc = json.load(fref)
            ref_start = ref_doc['athlete_frames'][0]['frame']
            our_start = athlete_frames[0]['frame']
            frame_offset = ref_start - our_start
            if frame_offset != 0:
                athlete_frames = [
                    {**e, 'frame': e['frame'] + frame_offset}
                    for e in athlete_frames
                ]
                print(f'  Frame offset +{frame_offset} applied '
                      f'(ref starts at {ref_start})')
        else:
            print(f'  No reference file found in {args.ref_dir}/, frames left as-is')

        year = int(seq[:4]) if seq[:4].isdigit() else None
        doc = {
            'player_id': seq,
            'year': year,
            'n_keypoints': n_kp,
            'athlete_frames': athlete_frames,
        }

        with open(out_path, 'w') as f:
            json.dump(doc, f, indent=2)

        print(f'  Exported {len(athlete_frames)} frames → {out_path}')
        n_ok += 1

    print(f'\nFinished. {n_ok} exported, {n_fail} failed.')


if __name__ == '__main__':
    main()
