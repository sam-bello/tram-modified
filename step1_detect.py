"""Step 1a: Detection + Segmentation + Tracking"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from glob import glob
from pycocotools import mask as masktool
from lib.pipeline import video2frames, detect_segment_track

video = sys.argv[1] if len(sys.argv) > 1 else './example_video.mov'
vis_mask = '--visualize_mask' in sys.argv

seq = os.path.basename(video).split('.')[0]
seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)

print('Extracting frames ...')
nframes = video2frames(video, img_folder)

print('Detect, Segment, and Track ...')
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25,
                                               min_size=100, save_vos=vis_mask)

np.save(f'{seq_folder}/boxes.npy', boxes_)
np.save(f'{seq_folder}/masks.npy', masks_)
np.save(f'{seq_folder}/tracks.npy', tracks_)
print(f'Detection complete! Saved to {seq_folder}/')
