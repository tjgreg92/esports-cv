#!/usr/bin/env python3
"""
Extract minimap crops from match video for YOLO training data.
"""

import cv2
from tqdm import tqdm

# Video and output paths
VIDEO_PATH = "data/video/match_01.mp4"
OUTPUT_DIR = "data/frames"

# Minimap crop coordinates
X = 55
Y = 798
W = 513
H = 247

# YOLO input size
YOLO_SIZE = (640, 640)

# Capture interval in seconds
INTERVAL_SECONDS = 5


def extract_training_data():
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps

    # Calculate frame interval
    frame_interval = int(fps * INTERVAL_SECONDS)

    # Calculate total frames to extract
    total_extractions = int(duration_seconds / INTERVAL_SECONDS)

    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration_seconds:.0f}s ({duration_seconds/60:.1f} min)")
    print(f"Extracting 1 frame every {INTERVAL_SECONDS}s ({total_extractions} frames)")
    print(f"Crop region: x={X}, y={Y}, w={W}, h={H}")
    print(f"Output size: {YOLO_SIZE[0]}x{YOLO_SIZE[1]}")
    print()

    frame_count = 0
    saved_count = 0

    with tqdm(total=total_extractions, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Check if this frame should be captured
            if frame_count % frame_interval == 0:
                # Crop the minimap region
                cropped = frame[Y:Y+H, X:X+W]

                # Resize to YOLO input size
                resized = cv2.resize(cropped, YOLO_SIZE, interpolation=cv2.INTER_LINEAR)

                # Save with zero-padded filename
                saved_count += 1
                output_path = f"{OUTPUT_DIR}/frame_{saved_count:04d}.jpg"
                cv2.imwrite(output_path, resized)

                pbar.update(1)

            frame_count += 1

    cap.release()

    print(f"\nDone! Saved {saved_count} frames to {OUTPUT_DIR}/")


if __name__ == "__main__":
    extract_training_data()
