#!/usr/bin/env python3
"""
audit_dataset.py

Audits all CSV files in the matches directory to identify:
- Ghost matches (too short)
- Glitched matches (impossible score jumps)
"""

import os
import glob
import pandas as pd
import numpy as np

# Configuration
MATCHES_DIR = "data/matches"
FPS = 60  # Frames per second
JUMP_THRESHOLD = 50  # Points per second that indicates a glitch


def audit_file(csv_path: str) -> dict:
    """
    Audit a single CSV file and return stats.
    """
    filename = os.path.basename(csv_path)

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        return {
            "filename": filename,
            "rows": 0,
            "duration_sec": 0,
            "duration_str": "ERROR",
            "final_score": f"ERROR: {e}",
            "max_jump_a": 0,
            "max_jump_b": 0,
            "has_glitch": True,
            "status": "ERROR"
        }

    rows = len(df)
    duration_sec = rows / FPS
    duration_min = int(duration_sec // 60)
    duration_remaining_sec = int(duration_sec % 60)
    duration_str = f"{duration_min}:{duration_remaining_sec:02d}"

    # Identify score columns
    if "team_a_score" in df.columns:
        score_a_col, score_b_col = "team_a_score", "team_b_score"
    elif "score_a" in df.columns:
        score_a_col, score_b_col = "score_a", "score_b"
    else:
        return {
            "filename": filename,
            "rows": rows,
            "duration_sec": duration_sec,
            "duration_str": duration_str,
            "final_score": "NO SCORE COLS",
            "max_jump_a": 0,
            "max_jump_b": 0,
            "has_glitch": True,
            "status": "NO COLS"
        }

    # Convert scores to numeric
    df[score_a_col] = pd.to_numeric(df[score_a_col], errors="coerce").fillna(0)
    df[score_b_col] = pd.to_numeric(df[score_b_col], errors="coerce").fillna(0)

    # Final scores
    final_a = int(df[score_a_col].iloc[-1])
    final_b = int(df[score_b_col].iloc[-1])
    final_score = f"{final_a} - {final_b}"

    # Calculate per-second jumps (sample every 60 frames)
    scores_a = df[score_a_col].values
    scores_b = df[score_b_col].values

    # Sample at 1-second intervals
    sample_indices = np.arange(0, len(scores_a), FPS)
    sampled_a = scores_a[sample_indices]
    sampled_b = scores_b[sample_indices]

    # Calculate diffs
    if len(sampled_a) > 1:
        diffs_a = np.diff(sampled_a)
        diffs_b = np.diff(sampled_b)
        max_jump_a = int(np.max(np.abs(diffs_a))) if len(diffs_a) > 0 else 0
        max_jump_b = int(np.max(np.abs(diffs_b))) if len(diffs_b) > 0 else 0
    else:
        max_jump_a = 0
        max_jump_b = 0

    # Determine if glitched
    has_glitch = max_jump_a > JUMP_THRESHOLD or max_jump_b > JUMP_THRESHOLD

    # Determine status
    if duration_sec < 60:
        status = "GHOST"
    elif has_glitch:
        status = "GLITCH"
    elif final_a < 200 and final_b < 200:
        status = "LOW SCORE"
    else:
        status = "OK"

    return {
        "filename": filename,
        "rows": rows,
        "duration_sec": duration_sec,
        "duration_str": duration_str,
        "final_score": final_score,
        "max_jump_a": max_jump_a,
        "max_jump_b": max_jump_b,
        "has_glitch": has_glitch,
        "status": status
    }


def main():
    print("=" * 100)
    print("  DATASET AUDIT")
    print("  Identifying Ghost & Glitched Matches")
    print("=" * 100)

    csv_files = sorted(glob.glob(os.path.join(MATCHES_DIR, "*.csv")))

    if not csv_files:
        print(f"\nNo CSV files found in {MATCHES_DIR}")
        return

    print(f"\nFound {len(csv_files)} files\n")

    # Audit all files
    results = [audit_file(f) for f in csv_files]

    # Print header
    print("-" * 100)
    print(f"{'Filename':<45} {'Duration':<10} {'Final Score':<15} {'Jump A':<8} {'Jump B':<8} {'Status':<10}")
    print("-" * 100)

    # Print results
    ghost_count = 0
    glitch_count = 0
    ok_count = 0

    for r in results:
        # Color coding via status
        status = r["status"]
        if status == "GHOST":
            ghost_count += 1
            status_display = "⚠️  GHOST"
        elif status == "GLITCH":
            glitch_count += 1
            status_display = "🔴 GLITCH"
        elif status == "LOW SCORE":
            status_display = "⚠️  LOW"
        elif status == "ERROR" or status == "NO COLS":
            status_display = "❌ ERROR"
        else:
            ok_count += 1
            status_display = "✅ OK"

        # Highlight jumps > threshold
        jump_a_str = str(r["max_jump_a"])
        jump_b_str = str(r["max_jump_b"])
        if r["max_jump_a"] > JUMP_THRESHOLD:
            jump_a_str = f"{r['max_jump_a']}!"
        if r["max_jump_b"] > JUMP_THRESHOLD:
            jump_b_str = f"{r['max_jump_b']}!"

        print(f"{r['filename']:<45} {r['duration_str']:<10} {r['final_score']:<15} {jump_a_str:<8} {jump_b_str:<8} {status_display:<10}")

    # Summary
    print("-" * 100)
    print(f"\n{'SUMMARY':=^100}")
    print(f"  Total files: {len(results)}")
    print(f"  ✅ OK: {ok_count}")
    print(f"  ⚠️  Ghost (< 60s): {ghost_count}")
    print(f"  🔴 Glitched (jump > {JUMP_THRESHOLD}/sec): {glitch_count}")
    print()

    # List problematic files
    if ghost_count > 0:
        print("  Ghost matches (too short):")
        for r in results:
            if r["status"] == "GHOST":
                print(f"    - {r['filename']} ({r['duration_str']})")

    if glitch_count > 0:
        print("\n  Glitched matches (impossible jumps):")
        for r in results:
            if r["status"] == "GLITCH":
                jumps = []
                if r["max_jump_a"] > JUMP_THRESHOLD:
                    jumps.append(f"Team A: +{r['max_jump_a']}/sec")
                if r["max_jump_b"] > JUMP_THRESHOLD:
                    jumps.append(f"Team B: +{r['max_jump_b']}/sec")
                print(f"    - {r['filename']} ({', '.join(jumps)})")


if __name__ == "__main__":
    main()
