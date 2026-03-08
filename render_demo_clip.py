#!/usr/bin/env python3
"""
render_demo_clip.py

Renders a 30-second demo clip (4:00-4:30 of the match) with a bold,
high-visibility overlay designed for projector presentation.

Also exports a thumbnail frame for the PowerPoint slide.

Usage:
    ./venv/bin/python3 render_demo_clip.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────

MATCH_DIR = "data/matches"
MATCH_FILES = [
    "match_jm752QUzGa8_map_1_hardpoint.csv",
    "match_jm752QUzGa8_map_4_hardpoint.csv",
    "match_zbm-dMLzUt8_map_1_hardpoint.csv",
]

VIDEO_PATH = "data/video/jm752QUzGa8.mp4"
VIDEO_FRAME_OFFSET = 203337

FPS = 59.94
CLIP_START_SEC = 240   # 4:00
CLIP_END_SEC   = 270   # 4:30

OUTPUT_VIDEO = "demo_clip.mp4"
OUTPUT_THUMB = "demo_thumbnail.png"

# ── Bold overlay design — high visibility for projectors ──────────────

BANNER_HEIGHT = 90          # Tall opaque banner
BAR_HEIGHT    = 20          # Thick bar
BAR_MARGIN    = 40          # Side margins for bar

# Colors (BGR)
BANNER_BG     = (25, 25, 25)        # Near-black
TEAM_A_GREEN  = (80, 200, 50)       # Bright green
TEAM_A_LIGHT  = (140, 230, 120)
TEAM_B_RED    = (60, 60, 230)       # Bright red
TEAM_B_LIGHT  = (140, 140, 240)
WHITE         = (255, 255, 255)

FEATURE_COLS = [
    "score_a", "score_b", "score_diff",
    "score_rate_a", "score_rate_b", "time_remaining_fraction",
    "player_count_map", "enemy_count_map", "player_advantage",
    "player_alive_rolling", "enemy_alive_rolling", "advantage_rolling",
]


# ── Model training (same as render_broadcast_ui.py) ───────────────────

def engineer_features(df):
    df = df.copy()
    df = df.rename(columns={"team_a_score": "score_a", "team_b_score": "score_b"})
    df["time_remaining"] = pd.to_numeric(df["time_remaining"], errors="coerce")
    total_time = df["time_remaining"].max()
    if pd.isna(total_time) or total_time == 0:
        total_time = 600.0
    df["time_remaining"] = df["time_remaining"].ffill().bfill()
    time_elapsed = (total_time - df["time_remaining"]).clip(lower=1)
    df["score_diff"] = df["score_a"] - df["score_b"]
    df["score_rate_a"] = df["score_a"] / time_elapsed
    df["score_rate_b"] = df["score_b"] / time_elapsed
    df["time_remaining_fraction"] = df["time_remaining"] / total_time
    df["player_advantage"] = df["player_count_map"] - df["enemy_count_map"]
    win = 1800
    df["player_alive_rolling"] = df["player_count_map"].rolling(win, center=True, min_periods=1).mean()
    df["enemy_alive_rolling"] = df["enemy_count_map"].rolling(win, center=True, min_periods=1).mean()
    df["advantage_rolling"] = df["player_advantage"].rolling(win, center=True, min_periods=1).mean()
    final = df.iloc[-1]
    df["winner"] = 1 if final["score_a"] > final["score_b"] else 0
    return df


def mirror_match(df):
    m = df.copy()
    m["score_a"], m["score_b"] = df["score_b"].values.copy(), df["score_a"].values.copy()
    m["score_diff"] = -df["score_diff"]
    m["score_rate_a"], m["score_rate_b"] = df["score_rate_b"].values.copy(), df["score_rate_a"].values.copy()
    m["player_count_map"], m["enemy_count_map"] = df["enemy_count_map"].values.copy(), df["player_count_map"].values.copy()
    m["player_advantage"] = -df["player_advantage"]
    m["player_alive_rolling"], m["enemy_alive_rolling"] = df["enemy_alive_rolling"].values.copy(), df["player_alive_rolling"].values.copy()
    m["advantage_rolling"] = -df["advantage_rolling"]
    m["winner"] = 1 - df["winner"]
    return m


def train_and_predict():
    print("Training model...")
    matches = []
    for fname in MATCH_FILES:
        df = pd.read_csv(os.path.join(MATCH_DIR, fname))
        feat = engineer_features(df)
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat.dropna(subset=FEATURE_COLS, inplace=True)
        matches.append(feat)

    train_parts = []
    for i in [0, 2]:
        train_parts.append(matches[i])
        train_parts.append(mirror_match(matches[i]))
    train_aug = pd.concat(train_parts, ignore_index=True)
    test = matches[1]

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    model.fit(train_aug[FEATURE_COLS], train_aug["winner"], verbose=False)

    probs = model.predict_proba(test[FEATURE_COLS])[:, 1]

    # Smooth
    smooth_window = int(FPS * 5)
    probs_smooth = pd.Series(probs).rolling(
        window=smooth_window, center=True, min_periods=1
    ).mean().values

    print(f"  Predictions ready: {len(probs):,} frames")
    return probs_smooth, test


# ── Bold overlay drawing ──────────────────────────────────────────────

def draw_bold_overlay(frame, prob_a, score_a, score_b):
    """
    High-visibility overlay:
    - Solid dark banner at bottom
    - Large score text + percentages
    - Thick colored probability bar
    """
    h, w = frame.shape[:2]
    prob_b = 1.0 - prob_a

    # Banner zone
    banner_top = h - BANNER_HEIGHT
    bar_left = BAR_MARGIN
    bar_right = w - BAR_MARGIN
    bar_w = bar_right - bar_left
    bar_top = h - 30
    split = int(bar_w * prob_a)

    # 1. Solid dark banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, banner_top), (w, h), BANNER_BG, -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

    # 2. Probability bar (thick, with rounded ends)
    # Team A (green)
    cv2.rectangle(frame,
                  (bar_left, bar_top),
                  (bar_left + split, bar_top + BAR_HEIGHT),
                  TEAM_A_GREEN, -1)
    # Team B (red)
    cv2.rectangle(frame,
                  (bar_left + split, bar_top),
                  (bar_right, bar_top + BAR_HEIGHT),
                  TEAM_B_RED, -1)

    # Rounded ends
    r = BAR_HEIGHT // 2
    cv2.circle(frame, (bar_left + r, bar_top + r), r, TEAM_A_GREEN, -1)
    if prob_a < 0.99:
        cv2.circle(frame, (bar_right - r, bar_top + r), r, TEAM_B_RED, -1)
    else:
        cv2.circle(frame, (bar_right - r, bar_top + r), r, TEAM_A_GREEN, -1)

    # 3. Text row: TEAM A  score  pct%  ···  WIN PROBABILITY  ···  pct%  score  TEAM B
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = banner_top + 38

    # Left: TEAM A  score  pct%
    x = bar_left
    cv2.putText(frame, "TEAM A", (x, text_y), font, 0.65, TEAM_A_LIGHT, 2, cv2.LINE_AA)
    x += 110

    score_a_str = str(int(score_a))
    cv2.putText(frame, score_a_str, (x, text_y), font, 0.9, WHITE, 2, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(score_a_str, font, 0.9, 2)
    x += sw + 18

    pct_a_str = f"{prob_a * 100:.0f}%"
    cv2.putText(frame, pct_a_str, (x, text_y), font, 0.7, WHITE, 2, cv2.LINE_AA)

    # Center: WIN PROBABILITY
    label = "WIN PROBABILITY"
    (lw, _), _ = cv2.getTextSize(label, font, 0.55, 1)
    cv2.putText(frame, label, ((w - lw) // 2, text_y), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Right: pct%  score  TEAM B (right-aligned)
    tb_label = "TEAM B"
    (tbw, _), _ = cv2.getTextSize(tb_label, font, 0.65, 2)
    cv2.putText(frame, tb_label, (bar_right - tbw, text_y), font, 0.65, TEAM_B_LIGHT, 2, cv2.LINE_AA)

    score_b_str = str(int(score_b))
    (sbw, _), _ = cv2.getTextSize(score_b_str, font, 0.9, 2)
    cv2.putText(frame, score_b_str, (bar_right - tbw - sbw - 18, text_y), font, 0.9, WHITE, 2, cv2.LINE_AA)

    pct_b_str = f"{prob_b * 100:.0f}%"
    (pbw, _), _ = cv2.getTextSize(pct_b_str, font, 0.7, 2)
    cv2.putText(frame, pct_b_str, (bar_right - tbw - sbw - pbw - 36, text_y), font, 0.7, WHITE, 2, cv2.LINE_AA)

    return frame


# ── Main render ───────────────────────────────────────────────────────

def main():
    probs, test_df = train_and_predict()

    scores_a = test_df["score_a"].values if "score_a" in test_df.columns else test_df["team_a_score"].values
    scores_b = test_df["score_b"].values if "score_b" in test_df.columns else test_df["team_b_score"].values

    # Frame range for clip
    frame_start = int(CLIP_START_SEC * FPS)
    frame_end = int(CLIP_END_SEC * FPS)
    clip_frames = frame_end - frame_start

    print(f"\nClip: {CLIP_START_SEC}s-{CLIP_END_SEC}s = frames {frame_start}-{frame_end} ({clip_frames} frames)")

    # Verify we have enough data
    if frame_end > len(probs):
        print(f"  ERROR: Match only has {len(probs)} frames, need {frame_end}")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {VIDEO_PATH}")
        sys.exit(1)

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    # Seek to clip start in source video
    source_frame = VIDEO_FRAME_OFFSET + frame_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    print(f"  Source video seek to frame {source_frame}")

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, vid_fps, (vid_w, vid_h))

    # Render with exponential smoothing
    prev_prob = probs[frame_start]
    thumbnail_saved = False

    print(f"  Rendering {clip_frames} frames...")

    for i in range(clip_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: Video ended at clip frame {i}")
            break

        match_frame = frame_start + i
        target_prob = probs[match_frame]

        # Smooth
        alpha = 0.05
        display_prob = prev_prob + (target_prob - prev_prob) * alpha
        prev_prob = display_prob
        display_prob = max(0.01, min(0.99, display_prob))

        frame = draw_bold_overlay(frame, display_prob, scores_a[match_frame], scores_b[match_frame])
        out.write(frame)

        # Save thumbnail at midpoint
        if not thumbnail_saved and i >= clip_frames // 2:
            cv2.imwrite(OUTPUT_THUMB, frame)
            thumbnail_saved = True
            print(f"  Thumbnail saved: {OUTPUT_THUMB}")

    cap.release()
    out.release()

    file_mb = os.path.getsize(OUTPUT_VIDEO) / (1024 * 1024)
    duration = clip_frames / vid_fps
    print(f"\n  Done! {OUTPUT_VIDEO} ({file_mb:.1f} MB, {duration:.1f}s)")
    print(f"  Thumbnail: {OUTPUT_THUMB}")


if __name__ == "__main__":
    main()
