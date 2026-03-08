#!/usr/bin/env python3
"""
clean_dataset.py

Cleans OCR-extracted telemetry data by:
1. Capping scores at 250 (Hardpoint max)
2. Enforcing monotonicity (scores never decrease)
3. Limiting jumps (max +10 per frame)
4. Deleting invalid non-Hardpoint matches
"""

import os
import glob
import pandas as pd
import numpy as np

# Configuration
MATCHES_DIR = "data/matches"
MAX_SCORE = 250
MAX_JUMP = 10
VALID_MIN_SCORE = 200  # Match must reach at least this score
VALID_MAX_SCORE = 260  # Allow slight OCR errors above 250


def sanity_check_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Apply sanity checks to score columns row-by-row.

    Rules:
    - Cap: If score > 250, replace with 250
    - Monotonicity: If score[t] < score[t-1], use score[t-1]
    - Jump Limit: If score[t] - score[t-1] > 10, use score[t-1] + 1
    """
    df = df.copy()

    # Identify score columns
    score_cols = []
    if "team_a_score" in df.columns:
        score_cols = ["team_a_score", "team_b_score"]
    elif "score_a" in df.columns:
        score_cols = ["score_a", "score_b"]

    if not score_cols:
        print("    WARNING: No score columns found!")
        return df

    for col in score_cols:
        # Convert to numeric, coercing errors
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Get values as array for faster processing
        scores = df[col].values.copy()

        # Process row by row
        for i in range(len(scores)):
            # Rule 1: Cap at MAX_SCORE
            if scores[i] > MAX_SCORE:
                scores[i] = MAX_SCORE

            if i > 0:
                prev_score = scores[i - 1]

                # Rule 2: Monotonicity - scores never decrease
                if scores[i] < prev_score:
                    scores[i] = prev_score

                # Rule 3: Jump Limit - max +10 per frame
                if scores[i] - prev_score > MAX_JUMP:
                    scores[i] = prev_score + 1

        df[col] = scores

    return df


def validate_hardpoint(df: pd.DataFrame, filename: str) -> bool:
    """
    Step 2: Validate that this is a legitimate Hardpoint match.

    Returns True if valid, False if should be deleted.
    """
    # Identify score columns
    if "team_a_score" in df.columns:
        score_a_col, score_b_col = "team_a_score", "team_b_score"
    elif "score_a" in df.columns:
        score_a_col, score_b_col = "score_a", "score_b"
    else:
        return False

    max_score_a = df[score_a_col].max()
    max_score_b = df[score_b_col].max()
    max_score = max(max_score_a, max_score_b)

    # Check if within valid range
    if max_score < VALID_MIN_SCORE:
        print(f"  DELETED: {filename} - Max Score: {max_score} (below {VALID_MIN_SCORE})")
        return False

    if max_score > VALID_MAX_SCORE:
        print(f"  DELETED: {filename} - Max Score: {max_score} (above {VALID_MAX_SCORE})")
        return False

    return True


def clean_dataset():
    """
    Main function to clean all CSV files.
    """
    print("=" * 60)
    print("  DATASET CLEANING")
    print("  Fixing OCR Hallucinations & Filtering Invalid Matches")
    print("=" * 60)

    csv_files = sorted(glob.glob(os.path.join(MATCHES_DIR, "*_hardpoint.csv")))

    if not csv_files:
        print(f"No *_hardpoint.csv files found in {MATCHES_DIR}")
        return

    print(f"\nFound {len(csv_files)} files to process\n")

    stats = {
        "processed": 0,
        "cleaned": 0,
        "deleted": 0,
        "scores_capped": 0,
        "monotonicity_fixes": 0,
        "jump_fixes": 0
    }

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)

        # Load the CSV
        df = pd.read_csv(csv_path, low_memory=False)
        original_len = len(df)

        # Track changes for this file
        if "team_a_score" in df.columns:
            score_cols = ["team_a_score", "team_b_score"]
        elif "score_a" in df.columns:
            score_cols = ["score_a", "score_b"]
        else:
            print(f"  SKIPPED: {filename} - No score columns")
            continue

        # Count issues before cleaning
        for col in score_cols:
            scores = pd.to_numeric(df[col], errors="coerce").fillna(0).values
            stats["scores_capped"] += np.sum(scores > MAX_SCORE)

            for i in range(1, len(scores)):
                if scores[i] < scores[i-1]:
                    stats["monotonicity_fixes"] += 1
                elif scores[i] - scores[i-1] > MAX_JUMP:
                    stats["jump_fixes"] += 1

        # Step 1: Apply sanity checks
        df_cleaned = sanity_check_scores(df)

        # Step 2: Validate Hardpoint
        is_valid = validate_hardpoint(df_cleaned, filename)

        if not is_valid:
            # Delete the file
            os.remove(csv_path)
            stats["deleted"] += 1
            continue

        # Step 3: Save cleaned version
        df_cleaned.to_csv(csv_path, index=False)
        stats["processed"] += 1
        stats["cleaned"] += 1

        # Get final scores for summary
        max_a = df_cleaned[score_cols[0]].max()
        max_b = df_cleaned[score_cols[1]].max()
        print(f"  CLEANED: {filename} - Final Scores: {int(max_a)}-{int(max_b)}")

    # Print summary
    print("\n" + "=" * 60)
    print("  CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Files processed: {stats['processed']}")
    print(f"  Files deleted: {stats['deleted']}")
    print(f"  Scores capped (>250): {stats['scores_capped']}")
    print(f"  Monotonicity fixes: {stats['monotonicity_fixes']}")
    print(f"  Jump limit fixes: {stats['jump_fixes']}")
    print(f"\n  Remaining valid matches: {stats['processed']}")

    return stats


def run_validation():
    """
    Step 4: Re-run validation after cleaning.
    """
    print("\n" + "=" * 60)
    print("  RE-RUNNING VALIDATION")
    print("=" * 60 + "\n")

    import subprocess
    result = subprocess.run(
        ["python", "validate_model.py"],
        capture_output=False,
        text=True
    )
    return result.returncode


if __name__ == "__main__":
    # Clean the dataset
    stats = clean_dataset()

    # Re-run validation
    if stats and stats["processed"] > 1:
        run_validation()
    else:
        print("\nNot enough valid matches remaining for validation.")
