"""Diagnostic: run detect_segment_track with GPU memory monitoring"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from glob import glob

imgfiles = sorted(glob('results/example_video/images/*.jpg'))
print(f'Total frames: {len(imgfiles)}')

# Check GPU memory before loading anything
print(f'GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')

from lib.pipeline.deva_track import get_deva_tracker, track_with_mask, flush_buffer
print(f'After DEVA import - allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')

from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
sam = sam.to('cuda')
predictor = SamPredictor(sam)
print(f'After SAM load - allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')

from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)
print(f'After ViTDet load - allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'After ViTDet load - reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB')

import cv2
from pycocotools import mask as masktool

# Process frames with memory monitoring
for t in range(len(imgfiles)):
    img_cv2 = cv2.imread(imgfiles[t])

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            det_out = detector(img_cv2)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.25)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            confs = det_instances.scores[valid_idx].cpu().numpy()
            boxes = np.hstack([boxes, confs[:, None]]) if len(boxes) > 0 else np.zeros((0,5))

    if t % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated()/1e9
        resv = torch.cuda.memory_reserved()/1e9
        print(f'Frame {t}: alloc={alloc:.2f}GB resv={resv:.2f}GB boxes={len(boxes)}')

    if t == 140:
        print(f'Passed frame 133! Stopping at 140.')
        break

print('Detection-only test passed!')
