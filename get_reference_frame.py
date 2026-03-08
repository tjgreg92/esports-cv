#!/usr/bin/env python3
"""
Extract a reference frame from the match video at the 5-minute mark.
"""

import cv2


def get_reference_frame():
    video_path = "data/video/match_01.mp4"
    output_path = "reference_map.jpg"

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get FPS to calculate frame number for 5-minute mark
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_time_seconds = 5 * 60  # 5 minutes
    target_frame = int(fps * target_time_seconds)

    # Skip to the 5-minute mark
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_path, frame)
        print("Frame saved!")
    else:
        print("Error: Could not read frame")

    cap.release()


if __name__ == "__main__":
    get_reference_frame()
