"""Exact replica of estimate_camera.py logic"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import argparse
import numpy as np
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track, visualize_tram
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world

file = './example_video.mov'
seq = 'example_video'
seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)

# Extract Frames (same as estimate_camera.py)
print('Extracting frames ...')
nframes = video2frames(file, img_folder)

# Detection + SAM + DEVA-Track-Anything
print('Detect, Segment, and Track ...')
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25,
                                               min_size=100, save_vos=False)
print(f'Detection complete! {len(boxes_)} frames processed')
