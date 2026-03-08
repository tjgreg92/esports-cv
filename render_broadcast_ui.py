#!/usr/bin/env python3
"""
render_broadcast_ui.py

Renders a broadcast-style presentation video with a live win probability
bar overlaid on match footage.

Pipeline:
  1. Train the XGBoost model (reuses logic from final_model_validation.py)
  2. Generate per-frame win probabilities for the hold-out match
  3. Render the source video with a dynamic win probability bar at the bottom

Usage:
  ./venv/bin/python3 render_broadcast_ui.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

MATCH_DIR = "data/matches"
MATCH_FILES = [
    "match_jm752QUzGa8_map_1_hardpoint.csv",  # Match 1 — Team B wins
    "match_jm752QUzGa8_map_4_hardpoint.csv",  # Match 2 — Team A wins (TEST)
    "match_zbm-dMLzUt8_map_1_hardpoint.csv",  # Match 3 — Team A wins
]

# Video source for the test match (Match 2)
VIDEO_PATH = "data/video/jm752QUzGa8.mp4"
VIDEO_FRAME_OFFSET = 203337  # Absolute frame where Match 2 (Hardpoint) starts in the video
OUTPUT_PATH = "presentation_demo.mp4"

FPS = 59.94

# UI layout — Option C: Full-width gradient + glow bar
BAR_HEIGHT = 8
BAR_MARGIN_SIDE = 100
OVERLAY_HEIGHT = 70          # Height of the gradient fade zone

# Colors (BGR — OpenCV uses BGR not RGB)
TEAM_A_COLOR = (94, 197, 34)      # #22c55e green
TEAM_A_LIGHT = (172, 239, 134)    # #86efac lighter green
TEAM_B_COLOR = (68, 68, 239)      # #ef4444 red
TEAM_B_LIGHT = (165, 165, 252)    # #fca5a5 lighter red
TEXT_COLOR = (255, 255, 255)
DIM_TEXT = (140, 140, 140)
TAG_COLOR = (90, 90, 90)


# ──────────────────────────────────────────────────────────────────────
# Step 1: Train model & generate probabilities
# ──────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "score_a", "score_b", "score_diff",
    "score_rate_a", "score_rate_b", "time_remaining_fraction",
    "player_count_map", "enemy_count_map", "player_advantage",
    "player_alive_rolling", "enemy_alive_rolling", "advantage_rolling",
]


def engineer_features(df):
    """Build features for a single match DataFrame."""
    df = df.copy()
    df = df.rename(columns={
        "team_a_score": "score_a",
        "team_b_score": "score_b",
    })

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

    win = 1800  # 30-second rolling window at ~60fps
    df["player_alive_rolling"] = df["player_count_map"].rolling(win, center=True, min_periods=1).mean()
    df["enemy_alive_rolling"] = df["enemy_count_map"].rolling(win, center=True, min_periods=1).mean()
    df["advantage_rolling"] = df["player_advantage"].rolling(win, center=True, min_periods=1).mean()

    final = df.iloc[-1]
    df["winner"] = 1 if final["score_a"] > final["score_b"] else 0

    return df


def mirror_match(df):
    """Swap Team A/B perspective for data augmentation."""
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
    """Train on matches 1 & 3, predict probabilities for match 2."""
    print("=" * 60)
    print("STEP 1: Training model & generating probabilities")
    print("=" * 60)

    matches = []
    for fname in MATCH_FILES:
        df = pd.read_csv(os.path.join(MATCH_DIR, fname))
        feat = engineer_features(df)
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat.dropna(subset=FEATURE_COLS, inplace=True)
        matches.append(feat)
        winner = "Team A" if feat["winner"].iloc[0] == 1 else "Team B"
        print(f"  Loaded {fname}: {len(feat):,} rows, winner={winner}")

    # Train on matches 0 and 2, test on match 1
    train_parts = []
    for i in [0, 2]:
        train_parts.append(matches[i])
        train_parts.append(mirror_match(matches[i]))
    train_aug = pd.concat(train_parts, ignore_index=True)

    test = matches[1]

    X_train = train_aug[FEATURE_COLS]
    y_train = train_aug["winner"]

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    # Predict per-frame probabilities for the test match
    X_test = test[FEATURE_COLS]
    probs = model.predict_proba(X_test)[:, 1]  # P(Team A wins)

    print(f"\n  Model trained on {len(train_aug):,} rows")
    print(f"  Predictions: {len(probs):,} frames")
    print(f"  P(Team A) range: {probs.min():.3f} – {probs.max():.3f}")
    print(f"  Final P(Team A): {probs[-1]:.3f}")

    return probs, test


# ──────────────────────────────────────────────────────────────────────
# Step 2 & 3: Render video with UI overlay
# ──────────────────────────────────────────────────────────────────────

def lerp(a, b, t):
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def draw_gradient_overlay(frame, top_y, bottom_y):
    """Draw a bottom-to-top gradient fade (black → transparent)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    zone_h = bottom_y - top_y
    for i in range(zone_h):
        # Alpha goes from 0 (transparent at top) to 0.85 (dark at bottom)
        alpha = 0.85 * (i / zone_h) ** 1.5
        y = top_y + i
        if 0 <= y < h:
            frame[y, :] = cv2.addWeighted(
                overlay[y, :], 1.0 - alpha,
                np.zeros_like(overlay[y, :]), 0.0,
                0,
            )
            # Darken by blending toward black
            frame[y, :] = (frame[y, :].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return frame


def draw_bar_glow(frame, bar_left, bar_top, bar_w, bar_h, prob_a):
    """Draw the thin probability bar with a soft glow effect."""
    h, w = frame.shape[:2]
    split = int(bar_w * prob_a)

    # --- Glow layer (drawn on a separate canvas, then blended) ---
    glow = np.zeros((h, w, 3), dtype=np.uint8)
    glow_radius = 12

    # Team A glow (green)
    cv2.rectangle(glow,
                  (bar_left, bar_top - glow_radius),
                  (bar_left + split, bar_top + bar_h + glow_radius),
                  TEAM_A_COLOR, -1)

    # Team B glow (red)
    cv2.rectangle(glow,
                  (bar_left + split, bar_top - glow_radius),
                  (bar_left + bar_w, bar_top + bar_h + glow_radius),
                  TEAM_B_COLOR, -1)

    # Blur for the glow effect
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=10, sigmaY=10)
    # Blend glow onto frame (additive-ish)
    cv2.addWeighted(frame, 1.0, glow, 0.35, 0, frame)

    # --- Solid bar on top ---
    # Team A fill (left → split)
    cv2.rectangle(frame,
                  (bar_left, bar_top),
                  (bar_left + split, bar_top + bar_h),
                  TEAM_A_LIGHT, -1)

    # Team B fill (split → right)
    cv2.rectangle(frame,
                  (bar_left + split, bar_top),
                  (bar_left + bar_w, bar_top + bar_h),
                  TEAM_B_LIGHT, -1)

    # Rounded ends
    radius = bar_h // 2
    cv2.circle(frame, (bar_left + radius, bar_top + radius), radius, TEAM_A_LIGHT, -1)
    if prob_a < 0.99:
        cv2.circle(frame, (bar_left + bar_w - radius, bar_top + radius), radius, TEAM_B_LIGHT, -1)
    else:
        cv2.circle(frame, (bar_left + bar_w - radius, bar_top + radius), radius, TEAM_A_LIGHT, -1)

    return frame


def draw_bar(frame, prob_a, score_a, score_b, time_str):
    """
    Option C — Full-width gradient with glow bar.

    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │  TEAM A   141   67%      WIN PROBABILITY      33%   106   TEAM B  │
    │  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
    │           XGBoost Model  ·  Minimap Telemetry + Scoreboard       │
    └─────────────────────────────────────────────────────────────────┘
    """
    h, w = frame.shape[:2]
    prob_b = 1.0 - prob_a

    # --- Geometry ---
    bar_left = BAR_MARGIN_SIDE
    bar_right = w - BAR_MARGIN_SIDE
    bar_w = bar_right - bar_left
    bar_top = h - 28          # Bar sits near the very bottom
    bar_h = BAR_HEIGHT        # Thin 8px bar
    text_row_y = bar_top - 12 # Scores + percentages row
    tag_y = h - 10            # Tagline below bar
    grad_top = h - OVERLAY_HEIGHT

    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- 1. Gradient background fade ---
    frame = draw_gradient_overlay(frame, grad_top, h)

    # --- 2. Glow bar ---
    frame = draw_bar_glow(frame, bar_left, bar_top, bar_w, bar_h, prob_a)

    # --- 3. Text row: TEAM A  score  pct%  ···  WIN PROBABILITY  ···  pct%  score  TEAM B ---

    # Left side: TEAM A  141  67%
    x_cursor = bar_left
    cv2.putText(frame, "TEAM A", (x_cursor, text_row_y),
                font, 0.48, TEAM_A_LIGHT, 1, cv2.LINE_AA)
    x_cursor += 82

    score_a_str = str(int(score_a))
    cv2.putText(frame, score_a_str, (x_cursor, text_row_y),
                font, 0.7, TEAM_A_LIGHT, 2, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(score_a_str, font, 0.7, 2)
    x_cursor += sw + 14

    pct_a_str = f"{prob_a * 100:.0f}%"
    cv2.putText(frame, pct_a_str, (x_cursor, text_row_y),
                font, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

    # Center: WIN PROBABILITY
    label = "WIN PROBABILITY"
    (lw, lh), _ = cv2.getTextSize(label, font, 0.4, 1)
    cv2.putText(frame, label, ((w - lw) // 2, text_row_y - 2),
                font, 0.4, DIM_TEXT, 1, cv2.LINE_AA)

    # Right side: pct%  score  TEAM B  (right-aligned)
    # Work backwards from bar_right
    team_b_label = "TEAM B"
    (tbw, _), _ = cv2.getTextSize(team_b_label, font, 0.48, 1)
    cv2.putText(frame, team_b_label, (bar_right - tbw, text_row_y),
                font, 0.48, TEAM_B_LIGHT, 1, cv2.LINE_AA)

    score_b_str = str(int(score_b))
    (sbw, _), _ = cv2.getTextSize(score_b_str, font, 0.7, 2)
    cv2.putText(frame, score_b_str, (bar_right - tbw - sbw - 14, text_row_y),
                font, 0.7, TEAM_B_LIGHT, 2, cv2.LINE_AA)

    pct_b_str = f"{prob_b * 100:.0f}%"
    (pbw, _), _ = cv2.getTextSize(pct_b_str, font, 0.55, 2)
    cv2.putText(frame, pct_b_str, (bar_right - tbw - sbw - pbw - 28, text_row_y),
                font, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

    # --- 4. Tagline below bar ---
    tag = "XGBoost Model  |  Minimap Telemetry + Scoreboard"
    (tagw, _), _ = cv2.getTextSize(tag, font, 0.32, 1)
    cv2.putText(frame, tag, ((w - tagw) // 2, tag_y),
                font, 0.32, TAG_COLOR, 1, cv2.LINE_AA)

    return frame


def render_video(probs, test_df):
    """
    Read the source video, overlay the win probability bar on each frame,
    and write to the output file.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Rendering broadcast UI")
    print("=" * 60)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {VIDEO_PATH}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    match_frames = len(probs)
    print(f"  Video: {vid_w}x{vid_h} @ {vid_fps:.2f}fps, {total_video_frames} total frames")
    print(f"  Match: {match_frames} frames starting at offset {VIDEO_FRAME_OFFSET}")
    print(f"  Output: {OUTPUT_PATH}")

    # Setup writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, vid_fps, (vid_w, vid_h))

    if not out.isOpened():
        print("  ERROR: Cannot create output video writer")
        sys.exit(1)

    # Seek to the match start
    cap.set(cv2.CAP_PROP_POS_FRAMES, VIDEO_FRAME_OFFSET)

    # Get score data for display
    scores_a = test_df["score_a"].values if "score_a" in test_df.columns else test_df["team_a_score"].values
    scores_b = test_df["score_b"].values if "score_b" in test_df.columns else test_df["team_b_score"].values

    # Smooth the probabilities with a rolling average for visual polish
    smooth_window = int(vid_fps * 5)  # 5-second smoothing
    probs_smooth = pd.Series(probs).rolling(
        window=smooth_window, center=True, min_periods=1
    ).mean().values

    # State for frame-level interpolation
    written = 0
    prev_prob = probs_smooth[0]

    print(f"\n  Rendering {match_frames} frames...")

    for i in range(match_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: Video ended at frame {i}")
            break

        # Current probability (already smoothed over 5s window)
        # Additional per-frame interpolation for buttery-smooth movement
        target_prob = probs_smooth[i]

        # Exponential smoothing for the visual bar (prevents any jitter)
        alpha = 0.05  # Lower = smoother bar movement
        display_prob = lerp(prev_prob, target_prob, alpha)
        prev_prob = display_prob

        # Clamp
        display_prob = max(0.01, min(0.99, display_prob))

        # Draw the UI
        frame = draw_bar(frame, display_prob, scores_a[i], scores_b[i], "")

        out.write(frame)
        written += 1

        # Progress
        if written % 3000 == 0 or written == match_frames:
            pct = written / match_frames * 100
            print(f"    {written:>6d}/{match_frames} frames ({pct:.1f}%)")

    cap.release()
    out.release()

    duration_sec = written / vid_fps
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\n  Done! Wrote {written} frames ({duration_sec:.1f}s)")
    print(f"  File: {OUTPUT_PATH} ({file_size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    probs, test_df = train_and_predict()
    render_video(probs, test_df)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Open with: open {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
