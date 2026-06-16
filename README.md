# TRAM — 3D Human Trajectory & Motion from Video

Fork of the official [TRAM](https://yufu-wang.github.io/tram4d/) implementation, extended for football video analysis:
- **PHALP-based tracking** (replaces DEVA) for more robust player re-identification
- **Field-mode camera alignment** using yard-line geometry (Experimental)
- **Batch keypoint export** to world-space 3D JSON

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Batch Export (export_keypoints.py)](#batch-export)
- [Keypoint Output Format](#keypoint-output-format)
- [Individual Scripts Reference](#individual-scripts-reference)
- [Training](#training)
- [Evaluation (EMDB)](#evaluation)

---

## Overview

The pipeline reconstructs world-space 3D human motion from a single monocular video in four steps:

```
Video
  │
  ▼
phalp_track.py        [4dhumans env]  ── human detection + tracking → masks_phalp.npy
  │
  ▼
estimate_camera.py    [tram env]      ── masked DROID-SLAM + metric depth → camera.npy
  │
  ▼
estimate_humans.py    [tram env]      ── HMR-VIMO pose/shape estimation → hps/hps_track_0.npy
  │
  ▼
export_keypoints.py   [tram env]      ── 45 world-space 3D joints → {seq}.json
```

All results are written to `results/{video_name}/`.

---

## Installation

### Prerequisites
- Anaconda or Miniconda
- CUDA-capable GPU (tested with CUDA 11.8 / 12.x)
- Windows or Linux (Windows path noted where relevant)

### 1. Clone the repo

```bash
git clone --recursive https://github.com/your-fork/tram
cd tram
```

### 2. Create the `tram` environment

```bash
conda create -n tram python=3.10 -y
conda activate tram
bash install.sh
```

Then compile the modified DROID-SLAM (required for masked SLAM):

```bash
cd thirdparty/DROID-SLAM
python setup.py install
cd ../..
```

**Windows note:** If you see `RuntimeError: Numpy is not available` or `TypeError: expected np.ndarray` when running scripts, pin NumPy and OpenCV to compatible versions:

```bash
pip install "numpy==1.26.4" "opencv-python==4.9.0.80" --force-reinstall
```

### 3. Create the `4dhumans` environment

The PHALP tracking step runs in a separate environment. Follow the [4D-Humans installation guide](https://github.com/shubham-goel/4D-Humans) to create the `4dhumans` conda environment, then install PHALP into it:

```bash
conda activate 4dhumans
pip install phalp[all]
```

---

## Data Setup

Register at [SMPLify](https://smplify.is.tue.mpg.de) and [SMPL](https://smpl.is.tue.mpg.de). The download script uses those credentials to fetch the SMPL body models.

```bash
conda activate tram
bash scripts/download_models.sh
```

This populates `data/` with:
- `data/smpl/` — SMPL body model (NEUTRAL, MALE, FEMALE)
- `data/pretrain/` — DROID-SLAM, SAM, DEVA, VIMO, and camera calibration checkpoints
- `example_video.mov` — sample video for testing

---

## Running the Pipeline

Run each step sequentially. Each step detects previously-generated outputs and skips completed steps automatically.

### Step 1 — Human tracking (4dhumans env)

```bash
conda run -n 4dhumans python scripts/phalp_track.py --video path/to/video.mp4
```

Outputs to `results/{seq}/`:
- `tracks_phalp.pkl` — per-frame track data
- `masks_phalp.npy` — binary masks used to exclude humans from SLAM

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | *(required)* | Path to the input video |
| `--max_age` | `50` | Frames to keep a track alive without a detection |

### Step 2 — Camera estimation (tram env)

```bash
conda run -n tram python scripts/estimate_camera.py --video path/to/video.mp4 [flags]
```

Outputs to `results/{seq}/`:
- `camera.npy` — per-frame camera rotation, translation, focal length, world alignment

| Flag | Description |
|------|-------------|
| `--static_camera` | Treat the camera as stationary (skips SLAM motion estimation) |
| `--field_mode` (Experimental: only works for some videos) | Use football yard lines (5 yd spacing) to compute metric scale and align the world coordinate frame so the field surface is Y=0 | 
| `--hoop_mode` (Experimental: only works for some videos) | Supplement yard-line scale with training hoops (requires `--field_mode`) |
| `--yard_line_align` (Experimental: only works for some videos) | Use yard-line vanishing point for world alignment instead of SPEC gravity (better for high-angle end-zone cameras) |
| `--visualize_mask` | Save DEVA masks as visualizations |

**Typical football usage:**
```bash
conda run -n tram python scripts/estimate_camera.py \
    --video Ravens_trimmed/2022_BARNO_AMARE_DL25.mp4 \
    --field_mode
```

### Step 3 — Human pose estimation (tram env)

```bash
conda run -n tram python scripts/estimate_humans.py --video path/to/video.mp4
```

Outputs to `results/{seq}/hps/`:
- `hps_track_0.npy` — pose (24×3×3 rotation matrices), shape (10 betas), translation per frame for the primary track
- Additional files for each tracked person

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | `./example_video.mov` | Path to the input video |
| `--max_humans` | `20` | Maximum number of people to reconstruct |

### Step 4 — Visualize (optional)

```bash
conda run -n tram python scripts/visualize_tram.py --video path/to/video.mp4 [--field_mode]
```

Produces `results/{seq}/tram_output.mp4`.

---

## Batch Export

`export_keypoints.py` runs the full 4-step pipeline for one video or a folder of videos, automatically skipping steps whose outputs already exist, and writes 3D keypoints to JSON.

### Single video

```bash
conda run -n tram python scripts/export_keypoints.py \
    --video Ravens_trimmed/2022_BARNO_AMARE_DL25.mp4 
```

### Folder of videos

```bash
conda run -n tram python scripts/export_keypoints.py \
    --video_dir Ravens_trimmed/ \
    --out_dir keypoints/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | — | Single video path (mutually exclusive with `--video_dir`) |
| `--video_dir` | — | Directory of `.mp4` files |
| `--out_dir` | `keypoints/` | Output directory for JSON files |
| `--field_mode` | off | Passed through to `estimate_camera.py` |
| `--hoop_mode` | off | Passed through to `estimate_camera.py` (requires `--field_mode`) |
| `--static_camera` | off | Passed through to `estimate_camera.py` |
| `--ref_dir` | `pose_json_3d_45_4dhumans/` | Directory of reference JSONs used to align starting frame numbers |
| `--device` | `cuda` | PyTorch device |

**Frame alignment:** If a reference JSON exists in `--ref_dir` for a given video, the output frame numbers are offset so the first frame matches the reference. This is used to align TRAM output with existing 4D-Humans annotations.

---

## Keypoint Output Format

Each JSON file contains 45 SMPL joints per frame in both image space (pixels) and world space (metres).

```json
{
  "player_id": "2022_BARNO_AMARE_DL25",
  "year": 2022,
  "n_keypoints": 45,
  "athlete_frames": [
    {
      "frame": 0,
      "keypoints":    [[u, v], ...],       // 45 × 2  image-space pixels
      "keypoints_3d": [[x, y, z], ...]     // 45 × 3  world-space metres
    },
    ...
  ]
}
```

**Coordinate conventions:**
- `keypoints` — pinhole projection into the original image, origin at top-left
- `keypoints_3d` — world frame set by field-mode alignment: field surface is Y=0, camera is at positive Y. Axes stored as `[z, y, x]` (depth-first order)
- When `--field_mode` is not used, the world frame origin and scale are SLAM-relative (metric scale from ZoeDepth, but no absolute ground reference)

**Joint ordering** follows the 45-joint raw SMPL-X convention (`default_smpl=True` in the SMPL wrapper).

---

## Individual Scripts Reference

| Script | Env | Purpose |
|--------|-----|---------|
| `scripts/phalp_track.py` | 4dhumans | PHALP human tracking |
| `scripts/estimate_camera.py` | tram | Masked DROID-SLAM + metric scale |
| `scripts/estimate_humans.py` | tram | HMR-VIMO pose & shape estimation |
| `scripts/visualize_tram.py` | tram | Render world-space output video |
| `scripts/export_keypoints.py` | tram | Full pipeline + JSON keypoint export |
| `scripts/emdb/run.sh` | tram | EMDB benchmark inference |
| `scripts/emdb/run_eval.py` | tram | EMDB evaluation metrics |
| `scripts/extract_bedlam_jpg.py` | tram | Extract frames from BEDLAM videos |
| `scripts/crop_datasets.py` | tram | Crop person bounding boxes for training |
---

## Acknowledgements

- [TRAM (original)](https://github.com/yufu-wang/tram) — Yufu Wang et al.
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) — baseline SLAM
- [ZoeDepth](https://github.com/isl-org/ZoeDepth) — metric depth
- [4D-Humans / HMR2.0](https://github.com/shubham-goel/4D-Humans) — backbone
- [PHALP](https://github.com/brjathu/PHALP) — 3D tracking
- [DEVA-Track-Anything](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) — video segmentation
- [Detectron2](https://github.com/facebookresearch/detectron2), [SAM](https://github.com/facebookresearch/segment-anything), [WHAM](https://github.com/yohanshin/WHAM), [BEDLAM](https://github.com/pixelite1201/BEDLAM), [EMDB](https://github.com/eth-ait/emdb)

---

## Citation

```bibtex
@article{wang2024tram,
  title={TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos},
  author={Wang, Yufu and Wang, Ziyun and Liu, Lingjie and Daniilidis, Kostas},
  journal={arXiv preprint arXiv:2403.17346},
  year={2024}
}
```
