"""Step 1b: DROID-SLAM camera estimation"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from glob import glob
from pycocotools import mask as masktool
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world

video = sys.argv[1] if len(sys.argv) > 1 else './example_video.mov'
static = '--static_camera' in sys.argv

seq = os.path.basename(video).split('.')[0]
seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

masks_ = np.load(f'{seq_folder}/masks.npy', allow_pickle=True)
masks = np.array([masktool.decode(m) for m in masks_])
masks = torch.from_numpy(masks)

print('Masked Metric SLAM ...')
cam_int, is_static = calibrate_intrinsics(img_folder, masks, is_static=static)
cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int, is_static=is_static)
wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(),
          'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
          'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}

np.save(f'{seq_folder}/camera.npy', camera)
print('SLAM complete!')
