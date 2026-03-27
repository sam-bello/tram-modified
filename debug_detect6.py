"""Minimal test: debug5 + argparse"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import argparse
import numpy as np
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track, visualize_tram
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov')
parser.add_argument("--static_camera", action='store_true')
parser.add_argument("--visualize_mask", action='store_true')
args = parser.parse_args()

file = args.video
seq = os.path.basename(file).split('.')[0]
seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)

print('Extracting frames ...')
nframes = video2frames(file, img_folder)

print('Detect, Segment, and Track ...')
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25,
                                               min_size=100, save_vos=args.visualize_mask)
print(f'Detection complete! {len(boxes_)} frames processed')
