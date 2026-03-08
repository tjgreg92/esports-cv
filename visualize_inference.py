#!/usr/bin/env python3
"""
Visualize YOLO inference on Call of Duty minimap.
Processes only the minimap region and draws colored bounding boxes.
"""

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import time

# Paths
MODEL_PATH = "runs/detect/runs/detect/cod_model_v1/weights/best.pt"
VIDEO_PATH = "data/video/match_01.mp4"
OUTPUT_PATH = "inference_demo.mp4"

# Minimap crop coordinates
CROP_X = 55
CROP_Y = 798
CROP_W = 513
CROP_H = 247

# Demo duration (seconds)
DEMO_DURATION = 180  # 3 minutes

# Colors (BGR format for OpenCV)
COLORS = {
    "player": (0, 255, 0),    # Green
    "enemy": (0, 0, 255),     # Red
}

# Class names
CLASS_NAMES = ["player", "enemy"]


def main():
    print("=" * 60)
    print("  MINIMAP INFERENCE VISUALIZATION")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("ERROR: Could not open video")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Limit to demo duration
    max_frames = int(fps * DEMO_DURATION)
    frames_to_process = min(total_frames, max_frames)

    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Processing: {frames_to_process} frames ({DEMO_DURATION}s demo)")
    print(f"  Minimap region: x={CROP_X}, y={CROP_Y}, w={CROP_W}, h={CROP_H}")

    # Setup video writer (output is minimap size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (CROP_W, CROP_H))

    print(f"\nProcessing video...")
    print(f"Output: {OUTPUT_PATH} ({CROP_W}x{CROP_H})")

    frame_count = 0

    with tqdm(total=frames_to_process, desc="Processing frames") as pbar:
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # Crop minimap region
            minimap = frame[CROP_Y:CROP_Y+CROP_H, CROP_X:CROP_X+CROP_W]

            # Run inference on minimap
            results = model.predict(
                minimap,
                imgsz=640,
                device="mps",
                verbose=False,
                conf=0.25
            )

            # Draw bounding boxes
            for result in results:
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get class name and color
                    class_name = CLASS_NAMES[class_id]
                    color = COLORS.get(class_name, (255, 255, 255))

                    # Draw rectangle
                    cv2.rectangle(minimap, (x1, y1), (x2, y2), color, 2)

                    # Draw label with confidence
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(minimap, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
                    cv2.putText(minimap, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Write frame
            out.write(minimap)
            frame_count += 1
            pbar.update(1)

            # Throttle to reduce CPU load (process ~30 fps worth of work)
            if frame_count % 2 == 0:
                time.sleep(0.001)

    # Cleanup
    cap.release()
    out.release()

    print(f"\n" + "=" * 60)
    print(f"  COMPLETE!")
    print(f"=" * 60)
    print(f"\nProcessed {frame_count} frames")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
