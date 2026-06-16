"""
Temporary script: backfill 2D keypoints into existing keypoints JSON files.

For each JSON in --kp_dir, loads the corresponding pipeline results
(camera.npy + hps/hps_track_0.npy) from results/<seq>/, projects the SMPL
joints into image space, and rewrites the JSON with:
  - "keypoints"    -> 2D image-space [u, v] (was the old 3D field)
  - "keypoints_3d" -> 3D world-space [x, y, z] (unchanged values, renamed)

Usage:
    conda run -n tram python scripts/backfill_2d_keypoints.py
    conda run -n tram python scripts/backfill_2d_keypoints.py --kp_dir keypoints/
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import json
import argparse
import numpy as np
import torch
from glob import glob

from lib.models.smpl import SMPL

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def backfill(json_path, smpl_model, device):
    with open(json_path) as f:
        doc = json.load(f)

    seq = doc['player_id']

    # Check if already backfilled
    if doc['athlete_frames'] and 'keypoints_3d' in doc['athlete_frames'][0]:
        print(f'  [{seq}] already has keypoints_3d, skipping')
        return True

    seq_folder = os.path.join(ROOT_DIR, 'results', seq)
    hps_path = os.path.join(seq_folder, 'hps', 'hps_track_0.npy')
    cam_path = os.path.join(seq_folder, 'camera.npy')

    if not os.path.exists(hps_path) or not os.path.exists(cam_path):
        print(f'  [{seq}] missing pipeline results, skipping')
        return False

    hps = np.load(hps_path, allow_pickle=True).item()
    cam = np.load(cam_path, allow_pickle=True).item()

    pred_rotmat = hps['pred_rotmat'].to(device)
    pred_shape  = hps['pred_shape'].to(device)
    pred_trans  = hps['pred_trans'].to(device)
    frame       = hps['frame']

    world_cam_R = torch.tensor(cam['world_cam_R'], dtype=torch.float32).to(device)
    world_cam_T = torch.tensor(cam['world_cam_T'], dtype=torch.float32).to(device)
    focal       = float(cam['img_focal'])
    cx, cy      = float(cam['img_center'][0]), float(cam['img_center'][1])

    mean_shape = pred_shape.mean(dim=0, keepdim=True).expand_as(pred_shape)

    with torch.no_grad():
        pred = smpl_model(
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=mean_shape,
            transl=pred_trans.squeeze(1),
            pose2rot=False,
            default_smpl=True,
        )

    # 2D projection (pinhole)
    joints_cam = pred.joints                                          # (N, 45, 3)
    u = focal * joints_cam[..., 0] / joints_cam[..., 2] + cx
    v = focal * joints_cam[..., 1] / joints_cam[..., 2] + cy
    joints_2d = torch.stack([u, v], dim=-1).cpu().numpy()            # (N, 45, 2)

    # 3D world-space
    cam_r = world_cam_R[frame]
    cam_t = world_cam_T[frame]
    joints_world = torch.einsum('bij,bnj->bni', cam_r, joints_cam) + cam_t[:, None]
    joints_world = joints_world.cpu().numpy()                        # (N, 45, 3)

    del pred_rotmat, pred_shape, pred_trans, mean_shape
    del world_cam_R, world_cam_T, cam_r, cam_t, pred, joints_cam, u, v
    torch.cuda.empty_cache()

    # Build a lookup from pipeline frame index → (kp_2d, kp_3d)
    frame_to_kps = {}
    for i, f in enumerate(frame.tolist()):
        frame_to_kps[int(f)] = (
            [[round(x, 2) for x in pt] for pt in joints_2d[i].tolist()],
            [[round(pt[2], 4), round(pt[1], 4), round(pt[0], 4)] for pt in joints_world[i].tolist()],
        )

    # Rewrite athlete_frames: rename old "keypoints" → "keypoints_3d", add new "keypoints"
    # The existing JSON keypoints are the 3D values (from the old export).
    # We replace them with freshly computed values to stay consistent.
    # Reconstruct the frame offset that export_keypoints.py may have applied:
    # offset = json_start - pipeline_start
    offset = 0
    if doc['athlete_frames'] and frame_to_kps:
        json_start = doc['athlete_frames'][0]['frame']
        pipeline_start = min(frame_to_kps.keys())
        offset = json_start - pipeline_start

    new_frames = []
    missing = 0
    for entry in doc['athlete_frames']:
        f = entry['frame']
        pipeline_frame = f - offset
        if pipeline_frame not in frame_to_kps:
            missing += 1
            new_frames.append(entry)
            continue
        kp_2d, kp_3d = frame_to_kps[pipeline_frame]
        new_frames.append({'frame': f, 'keypoints': kp_2d, 'keypoints_3d': kp_3d})

    if missing:
        print(f'  [{seq}] WARNING: {missing} frames had no matching pipeline data')

    doc['athlete_frames'] = new_frames
    with open(json_path, 'w') as f:
        json.dump(doc, f, indent=2)

    print(f'  [{seq}] updated {len(new_frames) - missing}/{len(new_frames)} frames -> {json_path}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kp_dir', default='keypoints',
                        help='Directory containing keypoint JSON files (default: keypoints/)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    kp_dir = os.path.join(ROOT_DIR, args.kp_dir)
    json_files = sorted(glob(os.path.join(kp_dir, '*.json')))
    if not json_files:
        print(f'No JSON files found in {kp_dir}')
        return

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Loading SMPL model on {device}...')
    smpl = SMPL().to(device).eval()

    n_ok = n_fail = 0
    for json_path in json_files:
        print(f'\n[{os.path.basename(json_path)}]')
        ok = backfill(json_path, smpl, device)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    del smpl
    torch.cuda.empty_cache()
    print(f'\nDone. {n_ok} updated, {n_fail} skipped/failed.')


if __name__ == '__main__':
    main()
