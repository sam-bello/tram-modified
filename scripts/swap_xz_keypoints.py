"""
Temporary script: swap x and z axes in keypoints_3d for existing keypoints JSONs.

Usage:
    python scripts/swap_xz_keypoints.py
    python scripts/swap_xz_keypoints.py --kp_dir keypoints/
"""

import json
import argparse
import os
from glob import glob

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kp_dir', default='keypoints',
                        help='Directory containing keypoint JSON files (default: keypoints/)')
    args = parser.parse_args()

    kp_dir = os.path.join(ROOT_DIR, args.kp_dir)
    json_files = sorted(glob(os.path.join(kp_dir, '*.json')))
    if not json_files:
        print(f'No JSON files found in {kp_dir}')
        return

    for json_path in json_files:
        with open(json_path) as f:
            doc = json.load(f)

        seq = doc['player_id']
        first = doc['athlete_frames'][0] if doc['athlete_frames'] else {}

        if 'keypoints_3d' not in first:
            print(f'[{seq}] no keypoints_3d field, skipping')
            continue

        for entry in doc['athlete_frames']:
            entry['keypoints_3d'] = [
                [pt[2], pt[1], pt[0]] for pt in entry['keypoints_3d']
            ]

        with open(json_path, 'w') as f:
            json.dump(doc, f, indent=2)

        print(f'[{seq}] swapped x/z -> {json_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
