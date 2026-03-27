"""Minimal pipeline launcher"""
import sys, os, subprocess
PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))

video = sys.argv[1] if len(sys.argv) > 1 else './example_video.mov'

# Step 1: Detection
print('=== Step 1: Detection ===')
r = subprocess.run([PYTHON, os.path.join(ROOT, 'step1_detect.py'), video], cwd=ROOT)
print(f'Detection exit code: {r.returncode}')
if r.returncode != 0:
    sys.exit(r.returncode)

# Step 2: SLAM
print('=== Step 2: SLAM ===')
r = subprocess.run([PYTHON, os.path.join(ROOT, 'step1_slam.py'), video], cwd=ROOT)
print(f'SLAM exit code: {r.returncode}')
