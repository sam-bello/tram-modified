"""Test if importing lib.camera (which sets multiprocessing start method) causes the crash"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import cv2
from glob import glob
from pycocotools import mask as masktool

# Import BOTH pipeline and camera modules - same as estimate_camera.py
from lib.pipeline import video2frames, detect_segment_track
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world

print(f'After all imports - GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')

imgfiles = sorted(glob('results/example_video/images/*.jpg'))
seq_folder = 'results/example_video'

print(f'Starting detect_segment_track on {len(imgfiles)} frames...')
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25,
                                               min_size=100, save_vos=False)
print(f'Detection complete! {len(boxes_)} boxes, {len(masks_)} masks')
