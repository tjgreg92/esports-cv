#!/usr/bin/env python3
"""
final_model_validation.py

End-to-end pipeline using full telemetry from the YOLOv8 minimap
detection + OCR scoreboard extraction:

  1. Load pre-processed match CSVs (with player/enemy minimap counts)
  2. Engineer features — scores, rates, AND spatial telemetry
  3. Train XGBClassifier on Matches 1 & 3, hold out Match 2
  4. Visualize predicted win probability vs actual scores
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

MATCH_DIR = "data/matches"
MATCH_FILES = [
    "match_jm752QUzGa8_map_1_hardpoint.csv",  # Match 1 — Team B wins (151-249)
    "match_jm752QUzGa8_map_4_hardpoint.csv",  # Match 2 — Team A wins (249-166)
    "match_zbm-dMLzUt8_map_1_hardpoint.csv",  # Match 3 — Team A wins (250-182)
]


# ──────────────────────────────────────────────────────────────────────
# Step 1: Load Matches
# ──────────────────────────────────────────────────────────────────────

def load_matches():
    """
    Load all match CSVs from the batch processing pipeline.
    These already contain per-frame data with minimap player counts.
    """
    print("=" * 60)
    print("STEP 1: Loading Match Data")
    print("=" * 60)

    matches = []
    for i, fname in enumerate(MATCH_FILES):
        path = os.path.join(MATCH_DIR, fname)
        df = pd.read_csv(path)

        # Standardize column names to match our pipeline
        df = df.rename(columns={
            "team_a_score": "score_a",
            "team_b_score": "score_b",
        })
        df["match_id"] = i

        # Determine winner from final scores
        last_valid = df.dropna(subset=["score_a", "score_b"]).iloc[-1]
        winner = "A" if last_valid["score_a"] > last_valid["score_b"] else "B"

        print(f"\n  Match {i+1}: {fname}")
        print(f"    Rows: {len(df):,}  |  "
              f"Frames: {df.frame.min()}-{df.frame.max()}")
        print(f"    Score A: {df.score_a.min():.0f}-{df.score_a.max():.0f}  |  "
              f"Score B: {df.score_b.min():.0f}-{df.score_b.max():.0f}")
        print(f"    Players on minimap: {df.player_count_map.min()}-{df.player_count_map.max()}  |  "
              f"Enemies on minimap: {df.enemy_count_map.min()}-{df.enemy_count_map.max()}")
        print(f"    Winner: Team {winner} "
              f"({last_valid.score_a:.0f}-{last_valid.score_b:.0f})")

        matches.append(df)

    return matches


# ──────────────────────────────────────────────────────────────────────
# Step 2: Feature Engineering
# ──────────────────────────────────────────────────────────────────────

def engineer_features(df):
    """
    Build features from a single match DataFrame.

    Scoreboard features:
      - score_diff:                Score A − Score B
      - score_rate_a/b:            Current score / time elapsed
      - time_remaining_fraction:   time_remaining / total match time

    Minimap telemetry features (the differentiator):
      - player_count_map:          Friendly players visible on minimap
      - enemy_count_map:           Enemy players visible on minimap
      - player_advantage:          player_count - enemy_count
      - player_alive_rolling:      Rolling avg of friendly players (30s window)
      - enemy_alive_rolling:       Rolling avg of enemy players (30s window)
      - advantage_rolling:         Rolling avg of player advantage (30s window)
    """
    df = df.copy()

    # --- Time features ---
    df["time_remaining"] = pd.to_numeric(df["time_remaining"], errors="coerce")
    total_time = df["time_remaining"].max()
    if pd.isna(total_time) or total_time == 0:
        total_time = 600.0

    df["time_remaining"] = df["time_remaining"].ffill().bfill()
    time_elapsed = total_time - df["time_remaining"]
    time_elapsed = time_elapsed.clip(lower=1)

    # --- Scoreboard features ---
    df["score_diff"] = df["score_a"] - df["score_b"]
    df["score_rate_a"] = df["score_a"] / time_elapsed
    df["score_rate_b"] = df["score_b"] / time_elapsed
    df["time_remaining_fraction"] = df["time_remaining"] / total_time

    # --- Minimap telemetry features ---
    df["player_advantage"] = df["player_count_map"] - df["enemy_count_map"]

    # Rolling averages over a 30-second window (~1800 frames at 59.94fps)
    win = 1800
    df["player_alive_rolling"] = (
        df["player_count_map"]
        .rolling(window=win, center=True, min_periods=1)
        .mean()
    )
    df["enemy_alive_rolling"] = (
        df["enemy_count_map"]
        .rolling(window=win, center=True, min_periods=1)
        .mean()
    )
    df["advantage_rolling"] = (
        df["player_advantage"]
        .rolling(window=win, center=True, min_periods=1)
        .mean()
    )

    # --- Label ---
    final = df.iloc[-1]
    winner = 1 if final["score_a"] > final["score_b"] else 0
    df["winner"] = winner

    return df


FEATURE_COLS = [
    # Scoreboard
    "score_a",
    "score_b",
    "score_diff",
    "score_rate_a",
    "score_rate_b",
    "time_remaining_fraction",
    # Minimap telemetry
    "player_count_map",
    "enemy_count_map",
    "player_advantage",
    "player_alive_rolling",
    "enemy_alive_rolling",
    "advantage_rolling",
]


def mirror_match(df):
    """
    Create a mirrored copy of a match by swapping Team A ↔ Team B.
    Swaps scores, rates, AND minimap perspectives.
    """
    mirrored = df.copy()

    # Swap scores
    mirrored["score_a"], mirrored["score_b"] = (
        df["score_b"].values.copy(),
        df["score_a"].values.copy(),
    )
    mirrored["score_diff"] = -df["score_diff"]
    mirrored["score_rate_a"], mirrored["score_rate_b"] = (
        df["score_rate_b"].values.copy(),
        df["score_rate_a"].values.copy(),
    )

    # Swap minimap perspectives
    mirrored["player_count_map"], mirrored["enemy_count_map"] = (
        df["enemy_count_map"].values.copy(),
        df["player_count_map"].values.copy(),
    )
    mirrored["player_advantage"] = -df["player_advantage"]
    mirrored["player_alive_rolling"], mirrored["enemy_alive_rolling"] = (
        df["enemy_alive_rolling"].values.copy(),
        df["player_alive_rolling"].values.copy(),
    )
    mirrored["advantage_rolling"] = -df["advantage_rolling"]

    mirrored["winner"] = 1 - df["winner"]
    return mirrored


# ──────────────────────────────────────────────────────────────────────
# Step 3: Training
# ──────────────────────────────────────────────────────────────────────

def train_model(train_dfs, test_df):
    """
    Train XGBClassifier on training matches, evaluate on held-out match.
    Each training match is augmented with its mirror to ensure both classes.
    """
    # Build augmented training set
    train_parts = []
    for tdf in train_dfs:
        train_parts.append(tdf)
        train_parts.append(mirror_match(tdf))
    train_aug = pd.concat(train_parts, ignore_index=True)

    X_train = train_aug[FEATURE_COLS]
    y_train = train_aug["winner"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["winner"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_prob, labels=[0, 1])

    # More meaningful metrics
    final_pred = y_prob[-1]
    final_correct = (final_pred > 0.5) == (y_test.iloc[-1] == 1)
    actual_winner = y_test.iloc[0]
    if actual_winner == 1:
        correct_mask = y_prob > 0.5
    else:
        correct_mask = y_prob < 0.5
    pct_correct = correct_mask.mean() * 100

    print("\n" + "=" * 60)
    print("STEP 3: Model Training & Evaluation")
    print("=" * 60)
    print(f"  Train matches: {len(train_dfs)} "
          f"({sum(len(d) for d in train_dfs):,} rows)")
    print(f"  Train augmented (+ mirrors): {len(train_aug):,} rows")
    print(f"  Test  match:                 {len(X_test):,} rows")
    print(f"  Class balance: {dict(y_train.value_counts())}")
    print(f"\n  Row-level accuracy:  {acc:.3f}")
    print(f"  Log Loss:            {loss:.4f}")
    print(f"  Final prediction:    {'Team A' if final_pred > 0.5 else 'Team B'} "
          f"({final_pred:.1%} Team A) — {'CORRECT' if final_correct else 'WRONG'}")
    print(f"  Correct winner call: {pct_correct:.1f}% of match duration")

    # Feature importance
    print("\n  Feature Importances:")
    importances = model.feature_importances_
    for feat, imp in sorted(
        zip(FEATURE_COLS, importances), key=lambda x: -x[1]
    ):
        bar = "█" * int(imp * 50)
        print(f"    {feat:<30s} {imp:.4f}  {bar}")

    return model, y_prob


# ──────────────────────────────────────────────────────────────────────
# Step 4: Visualization
# ──────────────────────────────────────────────────────────────────────

def plot_results(test_df, y_prob, match_label, fps=59.94, window_sec=30,
                 output_path="final_validation_chart.png"):
    """
    Two-panel chart:
      Top:    Actual scores (A vs B) over time
      Bottom: Smoothed win probability for Team A over time
    """
    # Convert frames to seconds from match start
    frame_start = test_df["frame"].iloc[0]
    time_sec = (test_df["frame"].values - frame_start) / fps

    # Smooth the probability with a rolling average
    sample_interval = np.median(np.diff(time_sec))
    window_samples = max(1, int(window_sec / sample_interval))
    prob_smooth = pd.Series(y_prob).rolling(
        window=window_samples, center=True, min_periods=1
    ).mean().values

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.08}
    )

    # --- Top: Scores ---
    ax1.plot(time_sec, test_df["score_a"].values, color="#e74c3c",
             linewidth=2, label="Team A")
    ax1.plot(time_sec, test_df["score_b"].values, color="#3498db",
             linewidth=2, label="Team B")
    ax1.axhline(250, color="gray", linestyle="--", alpha=0.4, label="Win (250)")
    ax1.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.set_ylim(-5, 270)
    ax1.set_title(f"{match_label}  —  Actual Scores vs. Predicted Win Probability",
                   fontsize=15, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.3)

    # --- Bottom: Win Probability ---
    ax2.fill_between(time_sec, 0.5, prob_smooth,
                     where=(prob_smooth >= 0.5),
                     color="#e74c3c", alpha=0.3, interpolate=True)
    ax2.fill_between(time_sec, 0.5, prob_smooth,
                     where=(prob_smooth < 0.5),
                     color="#3498db", alpha=0.3, interpolate=True)
    ax2.plot(time_sec, prob_smooth, color="black", linewidth=2)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax2.set_ylabel("Team A Win Probability", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Time (seconds from match start)", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.grid(True, alpha=0.3)

    # Add "Team A favored" / "Team B favored" labels
    ax2.text(time_sec[-1] * 0.98, 0.85, "Team A favored →",
             ha="right", fontsize=10, color="#e74c3c", alpha=0.7)
    ax2.text(time_sec[-1] * 0.98, 0.15, "Team B favored →",
             ha="right", fontsize=10, color="#3498db", alpha=0.7)

    fig.subplots_adjust(hspace=0.08, left=0.08, right=0.96, top=0.93, bottom=0.09)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # --- Step 1: Load all matches ---
    matches = load_matches()

    # --- Step 2: Feature Engineering ---
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)

    featured = []
    for i, mdf in enumerate(matches):
        feat = engineer_features(mdf)
        # Drop rows with inf/nan in features
        feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat.dropna(subset=FEATURE_COLS, inplace=True)
        featured.append(feat)

        winner = "Team A" if feat["winner"].iloc[0] == 1 else "Team B"
        print(f"  Match {i+1}: {len(feat):,} feature rows, winner = {winner}")

    print(f"\n  Feature set: {len(FEATURE_COLS)} features")
    print(f"  Scoreboard:  score_a, score_b, score_diff, score_rate_a/b, time_frac")
    print(f"  Telemetry:   player_count, enemy_count, advantage, rolling avgs")

    # Preview
    print(f"\n  Sample features (Match 1, row 10000):")
    sample = featured[0].iloc[10000]
    for col in FEATURE_COLS:
        print(f"    {col:<30s} {sample[col]:.3f}")

    # --- Step 3: Hold-out training ---
    # Train on matches 1 & 3 (different videos, different winners)
    # Test on match 2 (same video as match 1, but different map)
    train_set = [featured[0], featured[2]]  # Match 1 (B wins) + Match 3 (A wins)
    test_set = featured[1]                  # Match 2 (A wins) — held out

    print(f"\n  Train: Match 1 + Match 3 ({sum(len(d) for d in train_set):,} rows)")
    print(f"  Test:  Match 2 ({len(test_set):,} rows)")

    model, y_prob = train_model(train_set, test_set)

    # --- Step 4: Visualize ---
    print("\n" + "=" * 60)
    print("STEP 4: Visualization")
    print("=" * 60)
    plot_results(
        test_set, y_prob,
        match_label="Hold-Out Match",
        output_path="final_validation_chart.png",
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
