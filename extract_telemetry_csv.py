#!/usr/bin/env python3
"""
Extract telemetry data from Call of Duty match video.

Combines YOLO minimap detection with EasyOCR scoreboard reading
to create a comprehensive match telemetry CSV.
"""

import cv2
import pandas as pd
from ultralytics import YOLO
import easyocr
from tqdm import tqdm

# Paths
MODEL_PATH = "runs/detect/runs/detect/cod_model_v1/weights/best.pt"
VIDEO_PATH = "data/video/match_01.mp4"
OUTPUT_CSV = "match_telemetry.csv"

# Minimap crop coordinates
MINIMAP_X = 55
MINIMAP_Y = 798
MINIMAP_W = 513
MINIMAP_H = 247

# =============================================================
# OCR CROP REGIONS
# =============================================================
# Team A score box (x, y, w, h)
SCORE_A_CROP = (823, 90, 84, 45)

# Team B score box (x, y, w, h)
SCORE_B_CROP = (1033, 90, 84, 45)

# Game clock box (x, y, w, h)
TIME_CROP = (923, 73, 78, 30)
# =============================================================

# OCR runs every N frames (60 frames = 1 second at 60fps)
OCR_INTERVAL = 60

# Class names
CLASS_NAMES = ["player", "enemy"]


def crop_region(frame, region):
    """Crop a region from frame. Region is (x, y, w, h)."""
    x, y, w, h = region
    return frame[y:y+h, x:x+w]


def extract_number(reader, image):
    """Extract number from image using EasyOCR."""
    try:
        results = reader.readtext(image, allowlist='0123456789:')
        if results:
            # Return the text with highest confidence
            text = max(results, key=lambda x: x[2])[1]
            return text.strip()
    except Exception as e:
        pass
    return ""


def main():
    print("=" * 60)
    print("  MATCH TELEMETRY EXTRACTION")
    print("=" * 60)

    print(f"\nOCR Regions:")
    print(f"  Score A: x={SCORE_A_CROP[0]}, y={SCORE_A_CROP[1]}, w={SCORE_A_CROP[2]}, h={SCORE_A_CROP[3]}")
    print(f"  Score B: x={SCORE_B_CROP[0]}, y={SCORE_B_CROP[1]}, w={SCORE_B_CROP[2]}, h={SCORE_B_CROP[3]}")
    print(f"  Time:    x={TIME_CROP[0]}, y={TIME_CROP[1]}, w={TIME_CROP[2]}, h={TIME_CROP[3]}")

    # Initialize models
    print("\nLoading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("Initializing EasyOCR (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=True)

    # Open video
    print(f"\nOpening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  OCR interval: every {OCR_INTERVAL} frames ({OCR_INTERVAL/fps:.1f}s)")

    # Data storage
    telemetry_data = []

    # Cache for OCR values (persist between frames)
    last_time = ""
    last_score_a = ""
    last_score_b = ""

    frame_count = 0

    print("\nExtracting telemetry...")

    with tqdm(total=total_frames, desc="Processing") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # --- YOLO: Run every frame ---
            minimap = frame[MINIMAP_Y:MINIMAP_Y+MINIMAP_H, MINIMAP_X:MINIMAP_X+MINIMAP_W]

            results = model.predict(
                minimap,
                imgsz=640,
                device="mps",
                verbose=False,
                conf=0.25
            )

            # Count players and enemies
            player_count = 0
            enemy_count = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if CLASS_NAMES[class_id] == "player":
                        player_count += 1
                    elif CLASS_NAMES[class_id] == "enemy":
                        enemy_count += 1

            # --- OCR: Run every OCR_INTERVAL frames ---
            if frame_count % OCR_INTERVAL == 0:
                # Extract OCR regions
                score_a_img = crop_region(frame, SCORE_A_CROP)
                score_b_img = crop_region(frame, SCORE_B_CROP)
                time_img = crop_region(frame, TIME_CROP)

                # Run OCR
                last_time = extract_number(reader, time_img) or last_time
                last_score_a = extract_number(reader, score_a_img) or last_score_a
                last_score_b = extract_number(reader, score_b_img) or last_score_b

            # Store telemetry row
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

    # Save to CSV
    print(f"\nSaving telemetry to {OUTPUT_CSV}...")
    df = pd.DataFrame(telemetry_data)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n" + "=" * 60)
    print(f"  EXTRACTION COMPLETE!")
    print(f"=" * 60)
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"\nCSV Preview:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
