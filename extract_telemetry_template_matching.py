#!/usr/bin/env python3
"""
extract_telemetry_template_matching.py

Extracts score and time telemetry from match video using template matching.

Score reading: content-width digit counting + full-ROI enumeration (0-250).
  - Multi-template variants per digit for robustness across frames.
  - Coverage constraints prevent false positives from narrow templates.

Timer reading: fixed colon split + padded segment matching.

Post-processing: monotonicity enforcement, forward-fill gaps,
  jump limiting (Hardpoint scores only increase, max +5/sec).
"""

import os
import csv
import sys
import time
import cv2
import numpy as np

# Paths
VIDEO_PATH = "data/video/match_01.mp4"
SCORE_TEMPLATE_DIR = "data/templates/score_multi"
TIME_TEMPLATE_DIR = "data/templates/time_native"
OUTPUT_CSV = "clean_telemetry.csv"

# ROI coordinates (from calibration)
SCORE_A_ROI = [823, 90, 84, 45]
SCORE_B_ROI = [1033, 90, 84, 45]
TIMER_ROI = [923, 73, 78, 30]

# Timer config
TIMER_COLON_START = 24
TIMER_COLON_END = 31
TIMER_BINARY_THRESH = 150

# Score reading config
MAX_SCORE = 250
SCORE_JUMP_LIMIT = 5  # Max score increase per second


def load_score_templates(template_dir):
    """Load multi-variant score templates. Filenames: {digit}_{variant}.png"""
    templates = []
    for filename in sorted(os.listdir(template_dir)):
        if not filename.endswith(".png"):
            continue
        filepath = os.path.join(template_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        label = filename.split("_")[0]
        if label.isdigit():
            templates.append((str(int(label)), img))
    return templates


def load_time_templates(template_dir):
    """Load single time templates. Filenames: {digit}.png"""
    templates = []
    for filename in sorted(os.listdir(template_dir)):
        if not filename.endswith(".png"):
            continue
        filepath = os.path.join(template_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        name = filename.replace(".png", "")
        if name.isdigit():
            templates.append((str(int(name)), img))
    return templates


def get_digit_count(gray):
    """Determine digit count from content width in the score ROI."""
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    col_sums = np.sum(binary > 0, axis=0)
    content_cols = np.where(col_sums > 0)[0]
    if len(content_cols) == 0:
        return 0, 0, 0
    cs = content_cols[0]
    ce = content_cols[-1] + 1
    cw = ce - cs
    if cw <= 35:
        return 1, cs, ce
    elif cw <= 62:
        return 2, cs, ce
    else:
        return 3, cs, ce


def read_score(gray, templates):
    """
    Read score from a grayscale ROI crop.

    Strategy: determine digit count from content width, then enumerate all
    valid score values (0-250) with that many digits. For each candidate,
    find the best non-overlapping template placement and score by geometric
    mean of match confidences weighted by coverage.
    """
    n_digits, cs, ce = get_digit_count(gray)
    if n_digits == 0:
        return None
    cw = ce - cs
    roi_w = gray.shape[1]

    # Pre-compute: best match score per digit label at each column position
    digit_scores = {}
    digit_widths = {}
    for label, tmpl in templates:
        th, tw = tmpl.shape[:2]
        if th > gray.shape[0] or tw > gray.shape[1]:
            continue
        result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        scores = result.max(axis=0)
        n_pos = len(scores)
        if label not in digit_scores:
            digit_scores[label] = np.full(roi_w, -1.0)
            digit_widths[label] = np.zeros(roi_w, dtype=int)
        for i in range(n_pos):
            if scores[i] > digit_scores[label][i]:
                digit_scores[label][i] = scores[i]
                digit_widths[label][i] = tw

    best_value = None
    best_conf = -1

    for value in range(0, MAX_SCORE + 1):
        digits_str = str(value)
        if len(digits_str) != n_digits:
            continue
        if not all(d in digit_scores for d in digits_str):
            continue

        if n_digits == 1:
            d = digits_str[0]
            s, w = digit_scores[d], digit_widths[d]
            for c in range(max(0, cs - 3), min(roi_w, ce + 3)):
                if s[c] < 0.5 or w[c] == 0:
                    continue
                tw = int(w[c])
                cov = (min(c + tw, ce) - max(c, cs)) / cw if cw > 0 else 0
                if cov < 0.6:
                    continue
                score = s[c] * (0.8 + 0.2 * min(cov, 1.0))
                if score > best_conf:
                    best_conf = score
                    best_value = value

        elif n_digits == 2:
            d1, d2 = digits_str
            s1, w1 = digit_scores[d1], digit_widths[d1]
            s2, w2 = digit_scores[d2], digit_widths[d2]
            for c1 in range(max(0, cs - 3), min(roi_w, ce)):
                if s1[c1] < 0.5 or w1[c1] == 0:
                    continue
                tw1 = int(w1[c1])
                for gap in range(-1, 4):
                    c2 = c1 + tw1 + gap
                    if c2 < 0 or c2 >= roi_w or s2[c2] < 0.5 or w2[c2] == 0:
                        continue
                    tw2 = int(w2[c2])
                    span = (c2 + tw2) - c1
                    cov = span / cw if cw > 0 else 0
                    if cov < 0.70 or span > cw * 1.3:
                        continue
                    conf = (s1[c1] * s2[c2]) ** 0.5 * (0.8 + 0.2 * min(cov, 1.0))
                    if conf > best_conf:
                        best_conf = conf
                        best_value = value

        elif n_digits == 3:
            d1, d2, d3 = digits_str
            s1, w1 = digit_scores[d1], digit_widths[d1]
            s2, w2 = digit_scores[d2], digit_widths[d2]
            s3, w3 = digit_scores[d3], digit_widths[d3]
            for c1 in range(max(0, cs - 2), min(roi_w, ce - 10)):
                if s1[c1] < 0.4 or w1[c1] == 0:
                    continue
                tw1 = int(w1[c1])
                for gap1 in range(-1, 4):
                    c2 = c1 + tw1 + gap1
                    if c2 < 0 or c2 >= roi_w or s2[c2] < 0.4 or w2[c2] == 0:
                        continue
                    tw2 = int(w2[c2])
                    for gap2 in range(-1, 4):
                        c3 = c2 + tw2 + gap2
                        if c3 < 0 or c3 >= roi_w or s3[c3] < 0.4 or w3[c3] == 0:
                            continue
                        tw3 = int(w3[c3])
                        span = (c3 + tw3) - c1
                        cov = span / cw if cw > 0 else 0
                        if cov < 0.70 or span > cw * 1.3:
                            continue
                        conf = (s1[c1] * s2[c2] * s3[c3]) ** (1.0 / 3)
                        conf *= (0.8 + 0.2 * min(cov, 1.0))
                        if conf > best_conf:
                            best_conf = conf
                            best_value = value

    return best_value


def _segment_timer_region(gray, thresh=TIMER_BINARY_THRESH, min_width=3):
    """Segment a timer sub-region into digit columns."""
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    col_sums = np.sum(binary > 0, axis=0)
    has_content = col_sums > 0
    regions = []
    in_r = False
    start = 0
    for i in range(len(has_content)):
        if has_content[i] and not in_r:
            start = i
            in_r = True
        elif not has_content[i] and in_r:
            if i - start >= min_width:
                regions.append((start, i))
            in_r = False
    if in_r and len(has_content) - start >= min_width:
        regions.append((start, len(has_content)))
    return regions


def _match_padded_segment(region, seg_start, seg_end, templates, pad=2, threshold=0.6):
    """Match a padded segment against templates, return best (label, score)."""
    ps = max(0, seg_start - pad)
    pe = min(region.shape[1], seg_end + pad)
    seg = region[:, ps:pe]

    best_label = None
    best_score = 0
    for label, tmpl in templates:
        if tmpl.shape[0] > seg.shape[0] or tmpl.shape[1] > seg.shape[1]:
            continue
        result = cv2.matchTemplate(seg, tmpl, cv2.TM_CCOEFF_NORMED)
        val = result.max()
        if val > best_score:
            best_score = val
            best_label = label

    if best_score >= threshold:
        return best_label, best_score
    return None, 0


def read_timer(gray, templates):
    """
    Read game timer from a grayscale ROI crop.

    Splits at fixed colon position, reads minute digit and two seconds digits
    using padded segment matching.

    Returns total seconds or None.
    """
    min_region = gray[:, :TIMER_COLON_START]
    sec_region = gray[:, TIMER_COLON_END:]

    # Read minute digit
    min_segs = _segment_timer_region(min_region)
    if not min_segs:
        return None
    min_label, _ = _match_padded_segment(min_region, min_segs[-1][0], min_segs[-1][1], templates)
    if min_label is None:
        return None

    # Read seconds digits (expect exactly 2 segments)
    sec_segs = _segment_timer_region(sec_region)
    if len(sec_segs) != 2:
        return None

    tens_label, _ = _match_padded_segment(sec_region, sec_segs[0][0], sec_segs[0][1], templates)
    ones_label, _ = _match_padded_segment(sec_region, sec_segs[1][0], sec_segs[1][1], templates)

    if tens_label is None or ones_label is None:
        return None

    try:
        return int(min_label) * 60 + int(tens_label) * 10 + int(ones_label)
    except ValueError:
        return None


def post_process(results):
    """
    Apply temporal smoothing to raw extraction results.

    Rules (Hardpoint):
    1. Scores are monotonically non-decreasing (0 to 250)
    2. Outlier jumps (spike then return) are removed
    3. Forward-fill gaps from previous valid reading
    4. Cap scores at MAX_SCORE
    """
    n = len(results)
    if n == 0:
        return results

    # Pass 1: Cap scores and None out impossible values
    for i in range(n):
        fn, tv, sa, sb = results[i]
        if sa is not None and (sa < 0 or sa > MAX_SCORE):
            sa = None
        if sb is not None and (sb < 0 or sb > MAX_SCORE):
            sb = None
        if tv is not None and (tv < 0 or tv > 600):
            tv = None
        results[i] = (fn, tv, sa, sb)

    # Pass 2: Remove outlier spikes. A reading is an outlier if it jumps
    # significantly from both its neighbors. Look at windows of 5 readings.
    def remove_outliers(values, max_jump=20):
        """Replace outlier spikes with None. values is list of int|None."""
        cleaned = list(values)
        for i in range(1, len(cleaned) - 1):
            if cleaned[i] is None:
                continue
            # Find previous valid value
            prev_val = None
            for j in range(i - 1, max(i - 5, -1), -1):
                if cleaned[j] is not None:
                    prev_val = cleaned[j]
                    break
            # Find next valid value
            next_val = None
            for j in range(i + 1, min(i + 5, len(cleaned))):
                if cleaned[j] is not None:
                    next_val = cleaned[j]
                    break
            if prev_val is not None and next_val is not None:
                # If this value jumps far from both neighbors, and neighbors
                # are close to each other, it's an outlier
                diff_prev = cleaned[i] - prev_val
                diff_next = cleaned[i] - next_val
                neighbor_diff = abs(next_val - prev_val)
                if (diff_prev > max_jump and diff_next > max_jump
                        and neighbor_diff < max_jump):
                    cleaned[i] = None
        return cleaned

    score_a_vals = [r[2] for r in results]
    score_b_vals = [r[3] for r in results]

    score_a_vals = remove_outliers(score_a_vals, max_jump=20)
    score_b_vals = remove_outliers(score_b_vals, max_jump=20)

    for i in range(n):
        fn, tv, _, _ = results[i]
        results[i] = (fn, tv, score_a_vals[i], score_b_vals[i])

    # Pass 3: Enforce monotonicity - if a score decreases, reject it
    max_a = 0
    max_b = 0
    cleaned = []

    for i in range(n):
        fn, tv, sa, sb = results[i]

        if sa is not None:
            if sa >= max_a:
                max_a = sa
            else:
                sa = None

        if sb is not None:
            if sb >= max_b:
                max_b = sb
            else:
                sb = None

        cleaned.append((fn, tv, sa, sb))

    # Pass 4: Forward-fill gaps
    prev_a = None
    prev_b = None
    prev_t = None
    filled = []

    for fn, tv, sa, sb in cleaned:
        if sa is not None:
            prev_a = sa
        else:
            sa = prev_a

        if sb is not None:
            prev_b = sb
        else:
            sb = prev_b

        if tv is not None:
            prev_t = tv
        else:
            tv = prev_t

        filled.append((fn, tv, sa, sb))

    return filled


def main():
    print("=" * 60)
    print("  TELEMETRY EXTRACTION - Template Matching v2")
    print("=" * 60)

    print(f"\n  Score A ROI: {SCORE_A_ROI}")
    print(f"  Score B ROI: {SCORE_B_ROI}")
    print(f"  Timer ROI:   {TIMER_ROI}")

    # Load templates
    print("\nLoading templates...")
    score_templates = load_score_templates(SCORE_TEMPLATE_DIR)
    time_templates = load_time_templates(TIME_TEMPLATE_DIR)
    print(f"  Score: {len(score_templates)} templates from {SCORE_TEMPLATE_DIR}")
    print(f"  Time:  {len(time_templates)} templates from {TIME_TEMPLATE_DIR}")

    if not score_templates or not time_templates:
        print("ERROR: Missing templates.")
        sys.exit(1)

    # Open video
    print(f"\nOpening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = round(video_fps)
    total_samples = total_frames // sample_interval

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Sampling every {sample_interval} frames (~1 sec)")
    print(f"  Expected samples: {total_samples}")

    # Main extraction loop
    print(f"\nExtracting...")
    print("-" * 60)

    raw_results = []
    frame_num = 0
    sample_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % sample_interval == 0:
            sample_count += 1

            # Read scores
            x, y, w, h = SCORE_A_ROI
            gray_a = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            score_a = read_score(gray_a, score_templates)

            x, y, w, h = SCORE_B_ROI
            gray_b = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            score_b = read_score(gray_b, score_templates)

            # Read timer
            x, y, w, h = TIMER_ROI
            gray_t = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            time_val = read_timer(gray_t, time_templates)

            raw_results.append((frame_num, time_val, score_a, score_b))

            # Progress every 120 samples (~2 minutes of video)
            if sample_count % 120 == 0:
                elapsed = time.time() - start_time
                pct = (frame_num / total_frames) * 100
                rate = sample_count / elapsed if elapsed > 0 else 0
                sa_str = str(score_a) if score_a is not None else "?"
                sb_str = str(score_b) if score_b is not None else "?"
                t_str = str(time_val) if time_val is not None else "?"
                print(f"  [{pct:5.1f}%] Frame {frame_num:>7d} | "
                      f"Score: {sa_str:>3s} - {sb_str:>3s} | "
                      f"Time: {t_str:>4s} | "
                      f"({rate:.1f} samp/sec)")

        frame_num += 1

    cap.release()
    elapsed = time.time() - start_time

    # Raw stats
    n_total = len(raw_results)
    n_raw_a = sum(1 for _, _, a, _ in raw_results if a is not None)
    n_raw_b = sum(1 for _, _, _, b in raw_results if b is not None)
    n_raw_t = sum(1 for _, t, _, _ in raw_results if t is not None)

    print("-" * 60)
    print(f"\nRaw extraction: {n_total} samples in {elapsed:.1f}s")
    print(f"  Score A: {n_raw_a}/{n_total} ({n_raw_a/n_total*100:.1f}%)")
    print(f"  Score B: {n_raw_b}/{n_total} ({n_raw_b/n_total*100:.1f}%)")
    print(f"  Timer:   {n_raw_t}/{n_total} ({n_raw_t/n_total*100:.1f}%)")

    # Post-process
    print("\nApplying temporal smoothing...")
    cleaned = post_process(raw_results)

    n_clean_a = sum(1 for _, _, a, _ in cleaned if a is not None)
    n_clean_b = sum(1 for _, _, _, b in cleaned if b is not None)
    n_clean_t = sum(1 for _, t, _, _ in cleaned if t is not None)
    n_complete = sum(1 for _, t, a, b in cleaned
                     if t is not None and a is not None and b is not None)

    print(f"  Score A: {n_clean_a}/{n_total} ({n_clean_a/n_total*100:.1f}%)")
    print(f"  Score B: {n_clean_b}/{n_total} ({n_clean_b/n_total*100:.1f}%)")
    print(f"  Timer:   {n_clean_t}/{n_total} ({n_clean_t/n_total*100:.1f}%)")
    print(f"  Complete rows: {n_complete}/{n_total} ({n_complete/n_total*100:.1f}%)")

    # Write output
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_remaining", "score_a", "score_b"])
        for fn, tv, sa, sb in cleaned:
            writer.writerow([
                fn,
                tv if tv is not None else "",
                sa if sa is not None else "",
                sb if sb is not None else "",
            ])

    print(f"\nOutput: {OUTPUT_CSV}")
    print("Done!")


if __name__ == "__main__":
    main()
