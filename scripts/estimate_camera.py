import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

# Add CUDA DLL directories for Windows
if sys.platform == 'win32':
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    for p in [os.path.join(conda_prefix, 'bin'),
              os.path.join(conda_prefix, 'Library', 'bin'),
              os.path.join(conda_prefix, 'Lib', 'site-packages', 'numpy', '.libs')]:
        if os.path.isdir(p):
            os.add_dll_directory(p)

import torch
import argparse
import numpy as np

# NumPy 2.x + PyTorch 2.4 ABI workaround: torch.from_numpy / torch.as_tensor fail
# because PyTorch was compiled against NumPy 1.x. Replace both with a frombuffer
# path that avoids the strict C-level type check. Covers all lib calls from this script.
def _np_to_tensor(arr):
    _dtype_map = {
        'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64,
        'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64,
        'uint8': torch.uint8, 'uint16': torch.int16, 'bool': torch.bool,
    }
    arr = np.ascontiguousarray(arr)
    dt = _dtype_map[str(arr.dtype)]
    return torch.frombuffer(bytearray(arr.tobytes()), dtype=dt).reshape(arr.shape)

_orig_from_numpy = torch.from_numpy
def _from_numpy_compat(arr):
    try:
        return _orig_from_numpy(arr)
    except TypeError:
        return _np_to_tensor(arr)
torch.from_numpy = _from_numpy_compat

_orig_as_tensor = torch.as_tensor
def _as_tensor_compat(data, dtype=None, device=None):
    try:
        return _orig_as_tensor(data, dtype=dtype, device=device)
    except (TypeError, RuntimeError):
        if hasattr(data, 'dtype'):  # numpy array
            t = _np_to_tensor(data)
            if dtype is not None:
                t = t.to(dtype)
            if device is not None:
                t = t.to(device)
            return t
        raise
torch.as_tensor = _as_tensor_compat

# NumPy 2.x: numpy scalar types (int64, float64, etc.) no longer subclass Python
# Number, so torch.arange(np.int64) raises TypeError. Convert any numpy scalar
# arguments to their Python equivalents first.
def _to_py_scalar(x):
    return x.item() if hasattr(x, 'item') and not isinstance(x, torch.Tensor) else x

_orig_arange = torch.arange
def _arange_compat(*args, **kwargs):
    args = tuple(_to_py_scalar(a) for a in args)
    kwargs = {k: _to_py_scalar(v) for k, v in kwargs.items()}
    return _orig_arange(*args, **kwargs)
torch.arange = _arange_compat

import cv2
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track, visualize_tram
from lib.camera import run_metric_slam, calibrate_intrinsics, align_cam_to_world


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='./example_video.mov', help='input video')
parser.add_argument("--static_camera", action='store_true', help='whether the camera is static')
parser.add_argument("--visualize_mask", action='store_true', help='save deva vos for visualization')
parser.add_argument("--field_mode", action='store_true',
                    help='use football field yard lines (5 yd apart) for metric scale and world alignment')
parser.add_argument("--hoop_mode", action='store_true',
                    help='supplement yard-line scale with red training hoops (requires --field_mode)')
parser.add_argument("--yard_line_align", action='store_true',
                    help='use yard-line vanishing point for world alignment instead of SPEC '
                         '(useful for high-angle cameras where SPEC underestimates pitch)')
args = parser.parse_args()

# File and folders
file = args.video
root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

seq_folder = f'results/{seq}'
img_folder = f'{seq_folder}/images'
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)

# Progress log — written step-by-step so a native crash leaves a breadcrumb
_log_path = os.path.join(seq_folder, 'estimate_camera_progress.txt')
def _log(msg):
    print(msg, flush=True)
    with open(_log_path, 'a') as _f:
        _f.write(msg + '\n')
        _f.flush()

_log('START')

##### Extract Frames #####
_log('STEP extract_frames')
nframes = video2frames(file, img_folder)
_log(f'  frames extracted: {nframes}')

imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

##### Use PHALP masks if available, otherwise run DEVA #####
masks_phalp_path = f'{seq_folder}/masks_phalp.npy'

if os.path.exists(masks_phalp_path):
    _log('STEP load_phalp_masks (DEVA skipped)')
    phalp_arr = np.load(masks_phalp_path)        # (N, H, W) uint8
    masks = torch.frombuffer(bytearray(phalp_arr.tobytes()), dtype=torch.uint8).reshape(phalp_arr.shape)
    _log(f'  PHALP masks loaded: {masks.shape}')

    # Placeholder boxes/masks/tracks — estimate_humans.py uses tracks_phalp.pkl
    img0 = cv2.imread(imgfiles[0])
    H, W = img0.shape[:2]
    zero_rle = masktool.encode(np.asfortranarray(np.zeros((H, W), dtype=np.uint8)))
    masks_ = np.array([zero_rle] * nframes, dtype=object)
    boxes_ = np.array([np.zeros((0, 5), dtype=np.float32) for _ in range(nframes)],
                      dtype=object)
    tracks_ = np.array({}, dtype=object)

else:
    ##### Detection + SAM + DEVA-Track-Anything #####
    _log('STEP detect_segment_track')
    _log(f'  image files: {len(imgfiles)}')
    boxes_, masks_, tracks_ = detect_segment_track(imgfiles, seq_folder, thresh=0.25,
                                                   min_size=100, save_vos=args.visualize_mask)
    _log('  detect_segment_track done')

    _log('STEP prepare_masks')
    masks = np.array([masktool.decode(m) for m in masks_])
    masks = torch.frombuffer(bytearray(masks.tobytes()), dtype=torch.uint8).reshape(masks.shape)
    _log(f'  masks shape: {masks.shape}')

_log('STEP calibrate_intrinsics')
cam_int, is_static = calibrate_intrinsics(img_folder, masks, is_static=args.static_camera)
_log(f'  cam_int: {cam_int}, is_static: {is_static}')

if args.field_mode:
    _log('STEP run_field_metric_slam')
    from lib.pipeline.field_detection import run_field_metric_slam
    cam_R, cam_T, wd_cam_R, wd_cam_T, _, _ = run_field_metric_slam(
        img_folder, masks=masks, calib=cam_int, is_static=is_static,
        use_hoops=args.hoop_mode)
    spec_f = cam_int[0]
    _log('  run_field_metric_slam done')

else:
    _log('STEP run_metric_slam')
    cam_R, cam_T = run_metric_slam(img_folder, masks=masks, calib=cam_int, is_static=is_static)
    _log('  run_metric_slam done')
    if args.yard_line_align:
        _log('STEP align_cam_via_yard_lines')
        from lib.pipeline.field_detection import align_cam_via_yard_lines
        wd_cam_R, wd_cam_T, _ = align_cam_via_yard_lines(imgfiles, cam_R, cam_T, cam_int)
        spec_f = cam_int[0]
        _log('  align_cam_via_yard_lines done')
    else:
        _log('STEP align_cam_to_world')
        wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R, cam_T)
        _log('  align_cam_to_world done')

# field_mode applies a height shift so the field surface is at Y=0 and the
# camera is at positive Y.  Other paths leave the camera at Y≈0 with the
# ground below (ground_y unknown / NaN).
world_ground_y = 0.0 if args.field_mode else float('nan')

camera = {'pred_cam_R': cam_R.numpy(), 'pred_cam_T': cam_T.numpy(),
          'world_cam_R': wd_cam_R.numpy(), 'world_cam_T': wd_cam_T.numpy(),
          'img_focal': cam_int[0], 'img_center': cam_int[2:], 'spec_focal': spec_f,
          'field_mode': args.field_mode,
          'world_ground_y': world_ground_y}

np.save(f'{seq_folder}/camera.npy', camera)
np.save(f'{seq_folder}/boxes.npy', boxes_)
np.save(f'{seq_folder}/masks.npy', masks_)
np.save(f'{seq_folder}/tracks.npy', tracks_)

_log('DONE')
