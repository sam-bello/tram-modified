"""Test: same as debug_fault but WITHOUT importing lib.camera"""
import faulthandler
faulthandler.enable()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track, visualize_tram
# NOT importing lib.camera

file = './example_video.mov'
seq = 'example_video'
seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)

print('Extracting frames ...')
nframes = video2frames(file, img_folder)

print('Detect, Segment, and Track ...')
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25,
                                               min_size=100, save_vos=False)
print(f'Detection complete! {len(boxes_)} frames')

# NOW import lib.camera after detection
print('Importing SLAM modules...')
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world

print('Masked Metric SLAM ...')
masks = np.array([masktool.decode(m) for m in masks_])
masks = torch.from_numpy(masks)

cam_int, is_static = calibrate_intrinsics(img_folder, masks, is_static=False)
cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int, is_static=is_static)
wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)

camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(),
          'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
          'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f}

np.save(f'{seq_folder}/camera.npy', camera)
np.save(f'{seq_folder}/boxes.npy', boxes_)
np.save(f'{seq_folder}/masks.npy', masks_)
np.save(f'{seq_folder}/tracks.npy', tracks_)
print('Camera estimation complete!')
