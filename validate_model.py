#!/usr/bin/env python3
"""
validate_model.py

Validates win probability model on unseen match data using hold-out validation.
Trains on all matches except the last, then predicts on the held-out match.
"""

import os
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt

# Configuration
MATCHES_DIR = "data/matches"
GAME_MODE_CONFIG = {
    "hardpoint": {"max_score": 250}
}
ROLLING_WINDOW_SECONDS = 45
FPS = 60  # Approximate for rolling window calculation


def load_all_matches(matches_dir: str) -> pd.DataFrame:
    """
    Step 1: Load all hardpoint CSVs and tag with match_id.
    """
    csv_files = sorted(glob.glob(os.path.join(matches_dir, "*_hardpoint.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No *_hardpoint.csv files found in {matches_dir}")

    print(f"Found {len(csv_files)} hardpoint match files:")

    all_dfs = []
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        # Extract match_id from filename (e.g., "match_xKivpiOhumg_map_1_hardpoint.csv" -> "xKivpiOhumg_map_1")
        match_id = filename.replace("match_", "").replace("_hardpoint.csv", "")

        df = pd.read_csv(csv_path)
        df["match_id"] = match_id
        all_dfs.append(df)
        print(f"  - {filename}: {len(df)} rows -> match_id: {match_id}")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")

    return combined_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for win probability prediction.
    """
    df = df.copy()

    # Rename columns to standardized names if needed
    column_mapping = {
        "team_a_score": "score_a",
        "team_b_score": "score_b",
        "player_count_map": "player_count",
        "enemy_count_map": "enemy_count"
    }
    df = df.rename(columns=column_mapping)

    # Score difference
    df["score_diff"] = df["score_a"] - df["score_b"]

    # Normalized score difference (relative to max score)
    max_score = GAME_MODE_CONFIG["hardpoint"]["max_score"]
    df["score_diff_normalized"] = df["score_diff"] / max_score

    # Map control difference (player presence on minimap)
    df["map_control_diff"] = df["player_count"] - df["enemy_count"]

    # Time remaining (parse from MM:SS format if string)
    if df["time_remaining"].dtype == object:
        def parse_time(t):
            try:
                if pd.isna(t) or t == "":
                    return 0
                parts = str(t).split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                return float(t)
            except:
                return 0
        df["time_remaining"] = df["time_remaining"].apply(parse_time)

    # Fill NaN values
    df["time_remaining"] = df["time_remaining"].fillna(0)
    df["score_diff_normalized"] = df["score_diff_normalized"].fillna(0)
    df["map_control_diff"] = df["map_control_diff"].fillna(0)

    return df


def determine_winner(df: pd.DataFrame, match_id: str) -> int:
    """
    Determine who won a match based on final scores.
    Returns 1 if Team A won, 0 if Team B won.
    """
    match_data = df[df["match_id"] == match_id]
    final_score_a = match_data["score_a"].max()
    final_score_b = match_data["score_b"].max()

    return 1 if final_score_a > final_score_b else 0


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create win labels for each match.
    """
    df = df.copy()

    # Get unique match IDs
    match_ids = df["match_id"].unique()

    # Determine winner for each match
    winners = {mid: determine_winner(df, mid) for mid in match_ids}

    # Create win_label column
    df["win_label"] = df["match_id"].map(winners)

    print("\nMatch outcomes:")
    for mid, winner in winners.items():
        match_data = df[df["match_id"] == mid]
        final_a = match_data["score_a"].max()
        final_b = match_data["score_b"].max()
        print(f"  {mid}: Team A {final_a} - Team B {final_b} -> Winner: {'Team A' if winner == 1 else 'Team B'}")

    return df


def holdout_split(df: pd.DataFrame):
    """
    Step 2: Split data into train (all but last match) and test (last match).
    """
    match_ids = sorted(df["match_id"].unique())

    if len(match_ids) < 2:
        raise ValueError("Need at least 2 matches for hold-out validation")

    test_match_id = match_ids[-1]
    train_match_ids = match_ids[:-1]

    train_df = df[df["match_id"].isin(train_match_ids)].copy()
    test_df = df[df["match_id"] == test_match_id].copy()

    print(f"\n{'='*60}")
    print("HOLD-OUT SPLIT")
    print(f"{'='*60}")
    print(f"Training on Matches: {train_match_ids}")
    print(f"Testing on Match: {test_match_id}")
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    return train_df, test_df, test_match_id


def train_model(train_df: pd.DataFrame):
    """
    Step 3: Train XGBoost classifier.
    """
    features = ["score_diff_normalized", "map_control_diff", "time_remaining"]

    X_train = train_df[features]
    y_train = train_df["win_label"]

    print(f"\n{'='*60}")
    print("TRAINING MODEL")
    print(f"{'='*60}")
    print(f"Features: {features}")
    print(f"Training samples: {len(X_train)}")
    print(f"Class distribution: {y_train.value_counts().to_dict()}")

    # Check for class balance
    if len(y_train.unique()) < 2:
        print("\nWARNING: Only one class in training data!")
        print("Creating proxy target based on score_diff > 0 for training...")
        y_train = (train_df["score_diff"] > 0).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # Feature importance
    print("\nFeature Importance:")
    for feat, imp in zip(features, model.feature_importances_):
        print(f"  {feat}: {imp:.4f}")

    return model, features


def evaluate_and_plot(model, test_df: pd.DataFrame, features: list, test_match_id: str):
    """
    Step 4: Predict on test set and create the 'Truth' chart.
    """
    X_test = test_df[features]
    y_test = test_df["win_label"]

    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of Team A winning
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    if len(y_test.unique()) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Log Loss: {logloss:.4f}")

    # Actual winner
    actual_winner = y_test.iloc[0]
    final_score_a = test_df["score_a"].max()
    final_score_b = test_df["score_b"].max()
    print(f"\nActual Result: Team A {final_score_a} - Team B {final_score_b}")
    print(f"Actual Winner: {'Team A' if actual_winner == 1 else 'Team B'}")

    # Apply rolling average for smoothing
    rolling_window = ROLLING_WINDOW_SECONDS * FPS
    test_df = test_df.copy()
    test_df["win_prob_raw"] = y_pred_proba
    test_df["win_prob_smooth"] = test_df["win_prob_raw"].rolling(
        window=rolling_window, min_periods=1, center=True
    ).mean()

    # Create time axis (in seconds from start)
    test_df["time_seconds"] = np.arange(len(test_df)) / FPS

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot smoothed probability
    ax.plot(
        test_df["time_seconds"],
        test_df["win_prob_smooth"] * 100,
        color="#2ecc71" if actual_winner == 1 else "#e74c3c",
        linewidth=2,
        label="Team A Win Probability (Smoothed)"
    )

    # 50% reference line
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% Threshold")

    # Match end vertical line
    match_end_time = test_df["time_seconds"].max()
    ax.axvline(x=match_end_time, color="black", linestyle="-", linewidth=2, label="Match End")

    # Add final score annotation
    ax.annotate(
        f"Final: {int(final_score_a)}-{int(final_score_b)}\n{'Team A Wins' if actual_winner == 1 else 'Team B Wins'}",
        xy=(match_end_time, 50),
        xytext=(match_end_time - 60, 75 if actual_winner == 1 else 25),
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
    )

    # Styling
    ax.set_xlim(0, match_end_time + 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Team A Win Probability (%)", fontsize=12)
    ax.set_title(
        f"Win Probability Prediction - Hold-Out Validation\n"
        f"Match: {test_match_id} | Model trained on {len(test_df['match_id'].unique())} other matches",
        fontsize=14,
        fontweight="bold"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add score progression as secondary info
    ax2 = ax.twinx()
    ax2.fill_between(
        test_df["time_seconds"],
        test_df["score_a"],
        alpha=0.1,
        color="green",
        label="Team A Score"
    )
    ax2.fill_between(
        test_df["time_seconds"],
        test_df["score_b"],
        alpha=0.1,
        color="red",
        label="Team B Score"
    )
    ax2.set_ylabel("Score", fontsize=12, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, 300)

    plt.tight_layout()

    # Save chart
    output_path = f"validation_chart_match_{test_match_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved: {output_path}")

    plt.close()

    return test_df


def main():
    print("="*60)
    print("  WIN PROBABILITY MODEL VALIDATION")
    print("  Hold-Out Validation on Unseen Match")
    print("="*60)

    # Step 1: Load & Tag Data
    print("\n[Step 1] Loading match data...")
    df = load_all_matches(MATCHES_DIR)

    # Engineer features
    print("\n[Step 1b] Engineering features...")
    df = engineer_features(df)

    # Create labels
    print("\n[Step 1c] Creating win labels...")
    df = create_labels(df)

    # Step 2: Hold-Out Split
    print("\n[Step 2] Performing hold-out split...")
    train_df, test_df, test_match_id = holdout_split(df)

    # Step 3: Train Model
    print("\n[Step 3] Training XGBoost model...")
    model, features = train_model(train_df)

    # Step 4: Evaluate & Plot
    print("\n[Step 4] Evaluating on test match and generating chart...")
    result_df = evaluate_and_plot(model, test_df, features, test_match_id)

    print("\n" + "="*60)
    print("  VALIDATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
