"""
Trim Ravens videos to only include frames present in the corresponding
pose_json_3d_45_4dhumans JSON files. Saves trimmed videos to Ravens_trimmed/.

Frame numbers in the JSON are 0-indexed (matching the video2frames convention
in lib/pipeline/tools.py where count starts at 0).

Usage:
    # Batch mode: trim all videos that have a matching JSON in pose_json_3d_45_4dhumans/
    python trim_ravens_videos.py

    # Single video: auto-detect frame range from matching JSON
    python trim_ravens_videos.py --video "2022 NIC BARNO AMARE DL25.mp4"

    # Single video: manually specify start and end frames (0-indexed, inclusive)
    python trim_ravens_videos.py --video "2022 NIC BARNO AMARE DL25.mp4" --start 100 --end 500

Inputs:
    - Ravens/              Source video directory (.mp4 files)
    - pose_json_3d_45_4dhumans/   JSON files with athlete_frames defining frame ranges

Output:
    - Ravens_trimmed/      Trimmed videos (audio stripped)
"""

import json
import os
import re
import subprocess
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAVENS_DIR = os.path.join(SCRIPT_DIR, "Ravens")
JSON_DIR = os.path.join(SCRIPT_DIR, "pose_json_3d_45_4dhumans")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Ravens_trimmed")

# Find ffmpeg binary (imageio_ffmpeg ships one in the tram conda env)
FFMPEG_CANDIDATES = [
    r"C:\Users\007sb\.conda\envs\tram\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe",
]

def find_ffmpeg():
    for path in FFMPEG_CANDIDATES:
        if os.path.isfile(path):
            return path
    # Fall back to system ffmpeg
    return "ffmpeg"

FFMPEG = find_ffmpeg()


def extract_year_and_dl(filename):
    """Extract (year, dl_number) from a video or JSON filename.

    Video examples:  '2022 NIC BARNO AMARE DL25.mp4'
                     '2022 NIC BONITTO NIK DL 01.mp4'
    JSON examples:   '2022_BARNO_AMARE_DL25.json'
                     '2022_BONITTO_NIK_DL01.json'
    """
    stem = os.path.splitext(filename)[0]
    # Match year at the start
    year_match = re.match(r'^(\d{4})', stem)
    if not year_match:
        return None, None
    year = year_match.group(1)
    # Match DL number (e.g. DL25, DL 01, DL 24)
    dl_match = re.search(r'DL\s*(\d+)', stem, re.IGNORECASE)
    if not dl_match:
        return None, None
    dl_num = int(dl_match.group(1))
    return year, dl_num


def build_video_map():
    """Build a dict: (year, dl_num) -> video_filename"""
    video_map = {}
    for fname in os.listdir(RAVENS_DIR):
        if not fname.lower().endswith('.mp4'):
            continue
        year, dl_num = extract_year_and_dl(fname)
        if year and dl_num is not None:
            key = (year, dl_num)
            if key in video_map:
                print(f"WARNING: duplicate key {key} for {fname} and {video_map[key]}")
            video_map[key] = fname
    return video_map


def get_frame_range(json_path):
    """Return (min_frame, max_frame) from a JSON file's athlete_frames."""
    with open(json_path) as f:
        data = json.load(f)
    frames = [entry['frame'] for entry in data['athlete_frames']]
    return min(frames), max(frames)-50  # trim a bit of extra frames at the end to be safe (some JSONs have a few extra frames beyond the main action)


def trim_video(input_path, output_path, start_frame, end_frame):
    """Use ffmpeg to trim video to [start_frame, end_frame] (inclusive, 0-indexed)."""
    # Use select filter for frame-accurate trimming
    # -vsync vfr avoids duplicate frames when using select
    select_expr = f"between(n,{start_frame},{end_frame})"
    cmd = [
        FFMPEG,
        "-y",                          # overwrite output
        "-i", input_path,
        "-vf", f"select='{select_expr}',setpts=PTS-STARTPTS",
        "-vsync", "vfr",
        "-an",                         # drop audio (not needed for pose analysis)
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trim Ravens videos to pose JSON frame ranges.")
    parser.add_argument("--video", metavar="FILENAME",
                        help="Trim only this video file (filename only, e.g. '2022 NIC BARNO AMARE DL25.mp4')")
    parser.add_argument("--start", type=int, metavar="FRAME",
                        help="Start frame (0-indexed, inclusive). Requires --video.")
    parser.add_argument("--end", type=int, metavar="FRAME",
                        help="End frame (0-indexed, inclusive). Requires --video.")
    args = parser.parse_args()

    # Validate single-video mode args
    if args.video and (args.start is None) != (args.end is None):
        parser.error("--start and --end must be used together.")
    if (args.start is not None or args.end is not None) and not args.video:
        parser.error("--start/--end require --video.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Single video mode ---
    if args.video:
        input_path = os.path.join(RAVENS_DIR, args.video)
        if not os.path.isfile(input_path):
            print(f"ERROR: video not found: {input_path}")
            sys.exit(1)

        stem = os.path.splitext(args.video)[0]

        if args.start is not None:
            start_frame, end_frame = args.start, args.end
        else:
            # Look up frame range from matching JSON
            video_map = build_video_map()
            year, dl_num = extract_year_and_dl(args.video)
            if year is None:
                print("ERROR: could not parse year/DL from video filename.")
                sys.exit(1)
            key = (year, dl_num)
            json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
            matched_json = next(
                (f for f in json_files if extract_year_and_dl(f) == key), None
            )
            if matched_json is None:
                print(f"ERROR: no matching JSON found for key {key}. Use --start/--end to set frames manually.")
                sys.exit(1)
            start_frame, end_frame = get_frame_range(os.path.join(JSON_DIR, matched_json))

        output_fname = stem + "_trimmed.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_fname)
        print(f"Trimming frames {start_frame}-{end_frame}: {args.video} -> {output_fname}")
        success = trim_video(input_path, output_path, start_frame, end_frame)
        if success:
            print(f"Done. Output: {output_path}")
        else:
            print("Trimming failed.")
            sys.exit(1)
        return

    # --- Batch mode (original behaviour) ---
    video_map = build_video_map()
    json_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith('.json'))
    total = len(json_files)
    matched = 0
    skipped = 0

    for i, json_fname in enumerate(json_files, 1):
        year, dl_num = extract_year_and_dl(json_fname)
        if year is None:
            print(f"[{i}/{total}] SKIP (no year/DL): {json_fname}")
            skipped += 1
            continue

        key = (year, dl_num)
        if key not in video_map:
            print(f"[{i}/{total}] NO VIDEO FOUND for {json_fname} (key={key})")
            skipped += 1
            continue

        video_fname = video_map[key]
        input_path = os.path.join(RAVENS_DIR, video_fname)
        output_fname = os.path.splitext(json_fname)[0] + ".mp4"
        output_path = os.path.join(OUTPUT_DIR, output_fname)

        if os.path.exists(output_path):
            print(f"[{i}/{total}] EXISTS, skipping: {output_fname}")
            matched += 1
            continue

        json_path = os.path.join(JSON_DIR, json_fname)
        start_frame, end_frame = get_frame_range(json_path)

        print(f"[{i}/{total}] Trimming frames {start_frame}-{end_frame}: {video_fname} -> {output_fname}")
        success = trim_video(input_path, output_path, start_frame, end_frame)
        if success:
            matched += 1
        else:
            skipped += 1

    print(f"\nDone. Trimmed: {matched}, Skipped/errors: {skipped}, Total JSON: {total}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
