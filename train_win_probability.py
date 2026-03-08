#!/usr/bin/env python3
"""
Train a Win Probability model for Call of Duty matches.

Uses telemetry data (scores, time, map control) to predict
the probability of Team A winning at any point in the match.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# =============================================================
# CONFIGURATION
# =============================================================
GAME_MODE = 'Hardpoint'  # Options: 'Hardpoint', 'SnD', 'Overload'
WIN_SCORE = {
    'Hardpoint': 250,
    'SnD': 6,
    'Overload': 8
}

# Paths
TELEMETRY_CSV = "match_telemetry.csv"
OUTPUT_CHART = "win_probability_v2.png"

# Smoothing window for probability output (in seconds)
SMOOTHING_WINDOW = 30

# Features to use (normalized)
FEATURES = ['team_a_progress', 'team_b_progress', 'score_diff_normalized', 'map_control_diff']


def load_and_engineer_features(csv_path):
    """Load telemetry CSV and create engineered features."""
    print("=" * 60)
    print("  STEP 1: FEATURE ENGINEERING")
    print("=" * 60)

    print(f"\nGame Mode: {GAME_MODE}")
    print(f"Win Score: {WIN_SCORE[GAME_MODE]}")

    # Load CSV
    print(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    # Convert scores to numeric, handling empty strings
    df['team_a_score'] = pd.to_numeric(df['team_a_score'], errors='coerce')
    df['team_b_score'] = pd.to_numeric(df['team_b_score'], errors='coerce')
    df['time_remaining'] = pd.to_numeric(df['time_remaining'], errors='coerce')

    # Forward fill to handle OCR gaps
    print("\nApplying forward fill to handle missing OCR data...")
    df = df.ffill()

    # Fill any remaining NaN at the start with 0
    df = df.fillna(0)

    # Create engineered features
    print("\nCreating normalized features...")

    win_score = WIN_SCORE[GAME_MODE]

    # Progress features (score / win_score)
    df['team_a_progress'] = df['team_a_score'] / win_score
    df['team_b_progress'] = df['team_b_score'] / win_score

    # Normalized score difference
    df['score_diff_normalized'] = (df['team_a_score'] - df['team_b_score']) / win_score

    # Map control difference (already exists, just ensure it's there)
    df['map_control_diff'] = df['player_count_map'] - df['enemy_count_map']

    print(f"  team_a_progress: Team A score / {win_score}")
    print(f"  team_b_progress: Team B score / {win_score}")
    print(f"  score_diff_normalized: (Team A - Team B) / {win_score}")
    print(f"  map_control_diff: Players on map - Enemies on map")

    # Determine match outcome from final scores
    final_row = df.iloc[-1]
    team_a_final = final_row['team_a_score']
    team_b_final = final_row['team_b_score']

    team_a_won = 1 if team_a_final > team_b_final else 0
    winner = "Team A" if team_a_won else "Team B"

    print(f"\nMatch Result: {winner} wins ({int(team_a_final)} - {int(team_b_final)})")

    # For training with a single match, we create a proxy target:
    # "Is Team A currently in a winning position?" (score_diff > 0)
    # This gives us both classes (0 and 1) to train on
    df['team_a_wins'] = (df['score_diff_normalized'] > 0).astype(int)

    # For rows where score is tied, use map control as tiebreaker
    tied_mask = df['score_diff_normalized'] == 0
    df.loc[tied_mask, 'team_a_wins'] = (df.loc[tied_mask, 'map_control_diff'] > 0).astype(int)

    print(f"\nTraining target: 'Is Team A currently winning?'")
    print(f"  Team A ahead: {(df['team_a_wins'] == 1).sum()} frames")
    print(f"  Team B ahead: {(df['team_a_wins'] == 0).sum()} frames")

    # Show feature statistics
    print("\nFeature Statistics:")
    print(df[FEATURES + ['team_a_wins']].describe().round(3))

    return df


def train_model(df):
    """Train XGBoost classifier on the telemetry data."""
    print("\n" + "=" * 60)
    print("  STEP 2: TRAINING MODEL")
    print("=" * 60)

    # Filter out rows with missing features
    df_clean = df.dropna(subset=FEATURES)
    print(f"\nUsing {len(df_clean)} rows after removing missing values")

    # Prepare features and target
    X = df_clean[FEATURES]
    y = df_clean['team_a_wins']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"  Training set: {len(X_train)} rows")
    print(f"  Test set: {len(X_test)} rows")

    # Train XGBoost
    print("\nTraining XGBClassifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✓ Model trained!")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # Feature importance
    print("\nFeature Importance:")
    importance = dict(zip(FEATURES, model.feature_importances_))
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    return model


def create_win_probability_chart(df, model):
    """Generate smoothed win probability curve over the match timeline."""
    print("\n" + "=" * 60)
    print("  STEP 3: GENERATING WIN PROBABILITY CHART")
    print("=" * 60)

    # Clean data for prediction
    df_clean = df.dropna(subset=FEATURES).copy()

    # Get predictions for every frame
    X = df_clean[FEATURES]
    probabilities = model.predict_proba(X)[:, 1]  # Probability of Team A winning

    df_clean['win_prob'] = probabilities

    # Sample to 1 per second (every 60 frames at 60fps)
    df_clean['second'] = df_clean['frame'] // 60

    # Aggregate by second
    df_seconds = df_clean.groupby('second').agg({
        'win_prob': 'mean',
        'time_remaining': 'first',
        'score_diff_normalized': 'first',
        'team_a_score': 'first',
        'team_b_score': 'first'
    }).reset_index()

    # Apply smoothing
    print(f"\nApplying rolling average smoothing (window={SMOOTHING_WINDOW}s)...")
    df_seconds['win_prob_smooth'] = df_seconds['win_prob'].rolling(
        window=SMOOTHING_WINDOW, min_periods=1, center=True
    ).mean()

    print(f"Plotting {len(df_seconds)} data points (1 per second)")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot smoothed win probability
    ax.plot(df_seconds['second'], df_seconds['win_prob_smooth'] * 100,
            color='#3498db', linewidth=2, label='Team A Win Probability')

    # Add 50% reference line
    ax.axhline(y=50, color='#e74c3c', linestyle='--', linewidth=1.5,
               alpha=0.8, label='50% (Even odds)')

    # Fill above/below 50%
    ax.fill_between(df_seconds['second'], 50, df_seconds['win_prob_smooth'] * 100,
                    where=(df_seconds['win_prob_smooth'] * 100 >= 50),
                    color='#2ecc71', alpha=0.3, label='Team A favored')
    ax.fill_between(df_seconds['second'], 50, df_seconds['win_prob_smooth'] * 100,
                    where=(df_seconds['win_prob_smooth'] * 100 < 50),
                    color='#e74c3c', alpha=0.3, label='Team B favored')

    # Labels and title
    ax.set_xlabel('Match Time (seconds)', fontsize=12)
    ax.set_ylabel('Team A Win Probability (%)', fontsize=12)
    ax.set_title(f'Call of Duty {GAME_MODE} - Live Win Probability',
                 fontsize=14, fontweight='bold')

    # Set y-axis limits
    ax.set_ylim(0, 100)

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add final score annotation
    final_a = int(df_seconds['team_a_score'].iloc[-1])
    final_b = int(df_seconds['team_b_score'].iloc[-1])
    winner = "Team A" if final_a > final_b else "Team B"

    ax.annotate(f'Final: {final_a} - {final_b}\n{winner} Wins',
                xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Add game mode annotation
    ax.annotate(f'Mode: {GAME_MODE}\nWin Score: {WIN_SCORE[GAME_MODE]}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                ha='left', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    plt.tight_layout()

    # Save the plot
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {OUTPUT_CHART}")

    # Show some key moments
    print("\nKey Moments:")
    max_prob_idx = df_seconds['win_prob_smooth'].idxmax()
    min_prob_idx = df_seconds['win_prob_smooth'].idxmin()

    max_row = df_seconds.loc[max_prob_idx]
    min_row = df_seconds.loc[min_prob_idx]

    print(f"  Highest Team A win prob: {max_row['win_prob_smooth']*100:.1f}% at second {int(max_row['second'])}")
    print(f"  Lowest Team A win prob: {min_row['win_prob_smooth']*100:.1f}% at second {int(min_row['second'])}")

    return df_seconds


def main():
    print("\n" + "=" * 60)
    print("  WIN PROBABILITY MODEL TRAINING v2")
    print("  Call of Duty Esports Analytics")
    print("=" * 60)

    # Step 1: Load and engineer features
    df = load_and_engineer_features(TELEMETRY_CSV)

    # Step 2: Train the model
    model = train_model(df)

    # Step 3: Generate the win probability chart
    df_timeline = create_win_probability_chart(df, model)

    print("\n" + "=" * 60)
    print("  COMPLETE!")
    print("=" * 60)
    print("\nOutputs:")
    print(f"  - Win probability chart: {OUTPUT_CHART}")
    print("\nNext steps:")
    print("  - Add more match CSVs to improve the model")
    print("  - Use the model for live match predictions")


if __name__ == "__main__":
    main()
