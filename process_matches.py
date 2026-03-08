#!/usr/bin/env python3
"""
Process multiple Call of Duty match videos.

Downloads videos, extracts telemetry, intelligently segments by map,
and saves only Hardpoint matches as clean CSVs.
"""

import os
import subprocess
from pathlib import Path

import cv2
import pandas as pd
import easyocr
from ultralytics import YOLO
from tqdm import tqdm

# =============================================================
# CONFIGURATION - ADD YOUR YOUTUBE URLS HERE
# =============================================================
MATCH_URLS = [
    "https://www.youtube.com/watch?v=xKivpiOhumg",
    "https://www.youtube.com/watch?v=zbm-dMLzUt8",
    "https://www.youtube.com/watch?v=jm752QUzGa8&t=42s",
]

# Paths
VIDEO_DIR = Path("data/video")
OUTPUT_DIR = Path("data/matches")
MODEL_PATH = "runs/detect/runs/detect/cod_model_v1/weights/best.pt"

# Minimap crop coordinates
MINIMAP_X = 55
MINIMAP_Y = 798
MINIMAP_W = 513
MINIMAP_H = 247

# OCR crop regions (x, y, w, h)
SCORE_A_CROP = (823, 90, 84, 45)
SCORE_B_CROP = (1033, 90, 84, 45)
TIME_CROP = (923, 73, 78, 30)

# Processing settings
OCR_INTERVAL = 60  # Run OCR every N frames (60 = 1 second at 60fps)

# Segmentation settings
SCORE_RESET_THRESHOLD = 10  # Score drop that indicates a new map
MIN_SEGMENT_FRAMES = 3600   # Minimum frames for a valid segment (60 seconds at 60fps)

# Game mode detection
HARDPOINT_MIN_SCORE = 100   # If max score > 100, it's Hardpoint
SND_MAX_SCORE = 20          # If max score <= 20, it's SnD/Overload (discard)

# Class names for YOLO
CLASS_NAMES = ["player", "enemy"]


def download_video(url, video_dir):
    """Download a YouTube video if it doesn't already exist."""
    # Extract video ID from URL
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = url.split("/")[-1]

    output_path = video_dir / f"{video_id}.mp4"

    if output_path.exists():
        print(f"  ✓ Video already exists: {output_path.name}")
        return output_path

    print(f"  Downloading: {video_id}...")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
        "-o", str(output_path),
        "--merge-output-format", "mp4",
        url
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✓ Downloaded: {output_path.name}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Download failed: {e}")
        return None


def crop_region(frame, region):
    """Crop a region from frame. Region is (x, y, w, h)."""
    x, y, w, h = region
    return frame[y:y+h, x:x+w]


def extract_number(reader, image):
    """Extract number from image using EasyOCR."""
    try:
        results = reader.readtext(image, allowlist='0123456789:')
        if results:
            text = max(results, key=lambda x: x[2])[1]
            return text.strip()
    except Exception:
        pass
    return ""


def extract_telemetry(video_path, model, reader):
    """Extract telemetry data from a video file."""
    print(f"\n  Extracting telemetry from: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  ✗ Could not open video")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"    FPS: {fps:.2f}, Frames: {total_frames}")

    telemetry_data = []
    last_time = ""
    last_score_a = ""
    last_score_b = ""
    frame_count = 0

    with tqdm(total=total_frames, desc="    Processing", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO: Run every frame on minimap
            minimap = frame[MINIMAP_Y:MINIMAP_Y+MINIMAP_H, MINIMAP_X:MINIMAP_X+MINIMAP_W]

            results = model.predict(
                minimap,
                imgsz=640,
                device="mps",
                verbose=False,
                conf=0.25
            )

            player_count = 0
            enemy_count = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if CLASS_NAMES[class_id] == "player":
                        player_count += 1
                    elif CLASS_NAMES[class_id] == "enemy":
                        enemy_count += 1

            # OCR: Run every OCR_INTERVAL frames
            if frame_count % OCR_INTERVAL == 0:
                score_a_img = crop_region(frame, SCORE_A_CROP)
                score_b_img = crop_region(frame, SCORE_B_CROP)
                time_img = crop_region(frame, TIME_CROP)

                last_time = extract_number(reader, time_img) or last_time
                last_score_a = extract_number(reader, score_a_img) or last_score_a
                last_score_b = extract_number(reader, score_b_img) or last_score_b

            telemetry_data.append({
                "frame": frame_count,
                "time_remaining": last_time,
                "team_a_score": last_score_a,
                "team_b_score": last_score_b,
                "player_count_map": player_count,
                "enemy_count_map": enemy_count
            })

            frame_count += 1
            pbar.update(1)

    cap.release()

    df = pd.DataFrame(telemetry_data)

    # Convert scores to numeric
    df['team_a_score'] = pd.to_numeric(df['team_a_score'], errors='coerce')
    df['team_b_score'] = pd.to_numeric(df['team_b_score'], errors='coerce')

    # Forward fill missing values
    df = df.ffill().fillna(0)

    print(f"    ✓ Extracted {len(df)} frames of telemetry")

    return df


def detect_map_breaks(df):
    """Detect map breaks where scores reset to 0-0."""
    print(f"\n  Detecting map breaks...")

    # Calculate combined score
    df['combined_score'] = df['team_a_score'] + df['team_b_score']

    # Find frames where combined score drops significantly
    df['score_drop'] = df['combined_score'].diff()

    # Map breaks: score drops by more than threshold AND new score is near 0
    break_mask = (
        (df['score_drop'] < -SCORE_RESET_THRESHOLD) &
        (df['combined_score'] < 20)
    )

    # Get frame indices of breaks
    break_frames = df[break_mask]['frame'].tolist()

    # Add start and end frames
    segment_boundaries = [0] + break_frames + [len(df)]

    # Remove duplicates and sort
    segment_boundaries = sorted(set(segment_boundaries))

    print(f"    Found {len(segment_boundaries) - 1} potential map segments")
    print(f"    Break points at frames: {break_frames}")

    return segment_boundaries


def segment_dataframe(df, boundaries):
    """Split DataFrame into segments based on boundaries."""
    segments = []

    for i in range(len(boundaries) - 1):
        start_frame = boundaries[i]
        end_frame = boundaries[i + 1]

        segment = df[(df['frame'] >= start_frame) & (df['frame'] < end_frame)].copy()

        if len(segment) >= MIN_SEGMENT_FRAMES:
            # Reset frame numbers within segment
            segment['frame'] = segment['frame'] - segment['frame'].min()
            segments.append(segment)

    return segments


def classify_and_save_segments(segments, video_id, output_dir):
    """Classify segments by game mode and save Hardpoint matches."""
    print(f"\n  Classifying and saving segments...")

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    discarded_count = 0

    for i, segment in enumerate(segments):
        max_score_a = segment['team_a_score'].max()
        max_score_b = segment['team_b_score'].max()
        max_score = max(max_score_a, max_score_b)

        duration_seconds = len(segment) / 60  # Assuming 60fps

        if max_score > HARDPOINT_MIN_SCORE:
            # It's Hardpoint - save it
            map_num = saved_count + 1
            filename = f"match_{video_id}_map_{map_num}_hardpoint.csv"
            filepath = output_dir / filename

            segment.to_csv(filepath, index=False)

            print(f"    ✓ Saved: {filename}")
            print(f"      Duration: {duration_seconds:.0f}s, Max Score: {max_score:.0f}")

            saved_count += 1

        elif max_score <= SND_MAX_SCORE:
            # It's SnD/Overload - discard
            print(f"    ✗ Discarded segment {i+1}: SnD/Overload (max_score={max_score:.0f})")
            discarded_count += 1

        else:
            # Ambiguous - log but don't save
            print(f"    ? Skipped segment {i+1}: Ambiguous (max_score={max_score:.0f})")

    return saved_count, discarded_count


def process_single_video(url, model, reader, video_dir, output_dir):
    """Process a single video through the entire pipeline."""
    print(f"\n{'='*60}")
    print(f"  Processing: {url}")
    print(f"{'='*60}")

    # Step 1: Download
    video_path = download_video(url, video_dir)
    if video_path is None:
        return 0, 0

    # Extract video ID for naming
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = video_path.stem

    # Step 2: Extract telemetry
    df = extract_telemetry(video_path, model, reader)
    if df is None:
        return 0, 0

    # Step 3: Detect map breaks and segment
    boundaries = detect_map_breaks(df)
    segments = segment_dataframe(df, boundaries)

    print(f"    Valid segments after filtering: {len(segments)}")

    # Step 4: Classify and save
    saved, discarded = classify_and_save_segments(segments, video_id, output_dir)

    return saved, discarded


def main():
    print("\n" + "=" * 60)
    print("  BATCH MATCH PROCESSOR")
    print("  Call of Duty Esports Analytics")
    print("=" * 60)

    if not MATCH_URLS:
        print("\n⚠️  No URLs configured!")
        print("Add YouTube URLs to the MATCH_URLS list at the top of this script.")
        return

    print(f"\nURLs to process: {len(MATCH_URLS)}")

    # Create directories
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize models (once for all videos)
    print("\nInitializing models...")
    print("  Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("  Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=True)

    # Process each URL
    total_saved = 0
    total_discarded = 0

    for url in MATCH_URLS:
        saved, discarded = process_single_video(
            url, model, reader, VIDEO_DIR, OUTPUT_DIR
        )
        total_saved += saved
        total_discarded += discarded

    # Summary
    print("\n" + "=" * 60)
    print("  BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n  Videos processed: {len(MATCH_URLS)}")
    print(f"  Hardpoint maps saved: {total_saved}")
    print(f"  SnD/Overload maps discarded: {total_discarded}")
    print(f"\n  Output directory: {OUTPUT_DIR}")

    # List saved files
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    if csv_files:
        print(f"\n  Saved files:")
        for f in sorted(csv_files):
            print(f"    - {f.name}")


if __name__ == "__main__":
    main()
