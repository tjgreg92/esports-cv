#!/usr/bin/env python3
"""
extract_templates_from_video.py

Extracts digit templates directly from the video at native ROI resolution.
Scans through the video, crops the score/time ROIs, segments individual digits
using contour detection, and saves unique digit templates.

You label them afterward by renaming: digit_001.png -> 8.png, etc.
"""

import os
import cv2
import numpy as np

# Video and ROI config
VIDEO_PATH = "data/video/match_01.mp4"

# ROIs from earlier calibration
SCORE_A_ROI = [823, 90, 84, 45]
SCORE_B_ROI = [1033, 90, 84, 45]
TIMER_ROI = [923, 73, 78, 30]

# Output directories
SCORE_OUT = "data/templates/score_from_video"
TIME_OUT = "data/templates/time_from_video"

# Sampling: check every N seconds
SAMPLE_EVERY_SEC = 10

# Minimum contour size to be a digit (not noise)
MIN_DIGIT_HEIGHT = 10
MIN_DIGIT_WIDTH = 4


def segment_digits(gray_crop):
    """
    Segment individual digits from a grayscale ROI crop.
    Returns list of (x_pos, digit_image) sorted by x position.
    """
    # Threshold to isolate bright digits from dark background
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter noise
        if h < MIN_DIGIT_HEIGHT or w < MIN_DIGIT_WIDTH:
            continue

        # Extract the digit with a small padding
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray_crop.shape[1], x + w + pad)
        y2 = min(gray_crop.shape[0], y + h + pad)

        digit_img = gray_crop[y1:y2, x1:x2]
        digits.append((x, digit_img))

    # Sort left to right
    digits.sort(key=lambda d: d[0])
    return digits


def image_is_duplicate(new_img, existing_images, threshold=0.95):
    """Check if a digit image is too similar to existing templates."""
    for existing in existing_images:
        # Resize to same dimensions for comparison
        if existing.shape != new_img.shape:
            existing_resized = cv2.resize(existing, (new_img.shape[1], new_img.shape[0]))
        else:
            existing_resized = existing

        result = cv2.matchTemplate(new_img, existing_resized, cv2.TM_CCOEFF_NORMED)
        if result.max() > threshold:
            return True
    return False


def extract_from_roi(cap, roi, output_dir, total_frames, fps, label):
    """Extract unique digit templates from a specific ROI."""
    os.makedirs(output_dir, exist_ok=True)

    sample_interval = int(fps * SAMPLE_EVERY_SEC)
    unique_digits = []
    count = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_interval == 0:
            x, y, w, h = roi
            crop = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            digits = segment_digits(gray)

            for _, digit_img in digits:
                if not image_is_duplicate(digit_img, unique_digits):
                    unique_digits.append(digit_img)
                    count += 1
                    filepath = os.path.join(output_dir, f"digit_{count:03d}.png")
                    cv2.imwrite(filepath, digit_img)

        frame_num += 1

        if frame_num % (sample_interval * 10) == 0:
            pct = frame_num / total_frames * 100
            print(f"    [{pct:5.1f}%] {count} unique digits found from {label}")

    return count


def main():
    print("=" * 60)
    print("  TEMPLATE EXTRACTOR")
    print("  Extracting digit templates from video ROIs")
    print("=" * 60)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n  Video: {VIDEO_PATH}")
    print(f"  FPS: {fps:.2f}, Frames: {total_frames}")
    print(f"  Sampling every {SAMPLE_EVERY_SEC}s\n")

    # Extract score digits
    print("  Extracting SCORE digits...")
    score_count_a = extract_from_roi(cap, SCORE_A_ROI, SCORE_OUT, total_frames, fps, "Score A")
    score_count_b = extract_from_roi(cap, SCORE_B_ROI, SCORE_OUT, total_frames, fps, "Score B")

    # Extract time digits
    print("\n  Extracting TIME digits...")
    time_count = extract_from_roi(cap, TIMER_ROI, TIME_OUT, total_frames, fps, "Timer")

    cap.release()

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Score templates: {score_count_a + score_count_b} unique digits -> {SCORE_OUT}/")
    print(f"  Time templates:  {time_count} unique digits -> {TIME_OUT}/")
    print(f"\n  NEXT STEP:")
    print(f"  Open the output folders and rename each file:")
    print(f"    digit_001.png  ->  8.png  (if it looks like an 8)")
    print(f"    digit_002.png  ->  0.png  (if it looks like a 0)")
    print(f"    etc.")
    print(f"  For the time folder, also keep colon.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
