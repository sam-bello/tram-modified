"""Diagnostic: detect + SAM + DEVA to find crash source"""
import sys, os, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import cv2
from glob import glob
from pycocotools import mask as masktool
from lib.pipeline.tools import arrange_boxes

imgfiles = sorted(glob('results/example_video/images/*.jpg'))
print(f'Total frames: {len(imgfiles)}')

# Load DEVA
from lib.pipeline.deva_track import get_deva_tracker, track_with_mask, flush_buffer
print(f'After DEVA import - allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# Load SAM
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="data/pretrain/sam_vit_h_4b8939.pth")
sam = sam.to('cuda')
predictor = SamPredictor(sam)
print(f'After SAM load - allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# Load ViTDet
from lib.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
cfg_path = 'data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py'
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)
print(f'After ViTDet load - allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')

# DEVA tracker
vid_length = len(imgfiles)
seq_folder = 'results/example_video'
deva, result_saver = get_deva_tracker(vid_length, seq_folder)

autocast = torch.amp.autocast

mask = None
for t in range(len(imgfiles)):
    imgpath = imgfiles[t]
    img_cv2 = cv2.imread(imgpath)

    # Detection
    with torch.no_grad():
        with autocast('cuda'):
            det_out = detector(img_cv2)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.25)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            confs = det_instances.scores[valid_idx].cpu().numpy()
            boxes = np.hstack([boxes, confs[:, None]])
            boxes = arrange_boxes(boxes, mode='size', min_size=100)

    # SAM
    if len(boxes) > 0:
        with autocast('cuda'):
            predictor.set_image(img_cv2, image_format='BGR')
            bb = torch.tensor(boxes[:, :4]).cuda()
            bb = predictor.transform.apply_boxes_torch(bb, img_cv2.shape[:2])
            masks, scores, _ = predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=bb, multimask_output=False)
            scores = scores.cpu()
            masks = masks.cpu().squeeze(1)
            mask = masks.sum(dim=0)
    else:
        mask = np.zeros_like(mask) if mask is not None else np.zeros((img_cv2.shape[0], img_cv2.shape[1]))

    # DEVA tracking
    if len(boxes) > 0 and (boxes[:, -1] > 0.80).sum() > 0:
        track_valid = boxes[:, -1] > 0.80
        masks_track = masks[track_valid]
        scores_track = scores[track_valid]
    else:
        masks_track = torch.zeros([1, img_cv2.shape[0], img_cv2.shape[1]])
        scores_track = torch.zeros([1])

    with autocast('cuda'):
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        track_with_mask(deva, masks_track, scores_track, img_rgb,
                        imgpath, result_saver, t, False)

    if t % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated()/1e9
        resv = torch.cuda.memory_reserved()/1e9
        print(f'Frame {t}: alloc={alloc:.2f}GB resv={resv:.2f}GB boxes={len(boxes)}')

print('Full pipeline test passed!')
