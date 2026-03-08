#!/usr/bin/env python3
"""
generate_pipeline_diagram.py

Generates a clean pipeline architecture diagram for the LaTeX report.
Output fits in a single column of a two-column article layout (~3.25 inches).

Usage:
    python generate_pipeline_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import shutil

OUTPUT = os.path.join("report", "figures", "Pipeline_diagram.png")
ROOT_COPY = "Pipeline_diagram.png"


def draw_box(ax, x_center, y_center, width, height, label, sublabel,
             facecolor, edgecolor):
    """Draw a rounded rectangle with centered two-line text."""
    x = x_center - width / 2
    y = y_center - height / 2
    box = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x_center, y_center + 0.11, label,
            ha='center', va='center', fontsize=7.5,
            fontweight='bold', fontfamily='sans-serif', color='#212121',
            zorder=3)
    ax.text(x_center, y_center - 0.11, sublabel,
            ha='center', va='center', fontsize=6,
            fontfamily='sans-serif', fontstyle='italic', color='#555555',
            zorder=3)


def draw_arrow(ax, x_from, y_from, x_to, y_to):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=(x_to, y_to), xytext=(x_from, y_from),
        arrowprops=dict(
            arrowstyle='->', color='#616161', lw=1.3,
            shrinkA=0, shrinkB=0,
            connectionstyle='arc3,rad=0',
        ),
        zorder=1,
    )


def main():
    fig, ax = plt.subplots(figsize=(3.5, 5.6))
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 5.6)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Box dimensions
    full_w = 2.8
    half_w = 1.28
    box_h = 0.55

    # X positions
    cx = 1.75                       # center for full-width boxes
    cx_left = 0.87                  # center for left parallel box
    cx_right = 2.63                 # center for right parallel box

    # Y positions (top to bottom)
    y1 = 5.05   # Broadcast Video
    y2 = 4.10   # Minimap + Scoreboard (parallel)
    y3 = 3.15   # Feature Engineering
    y4 = 2.20   # XGBoost
    y5 = 1.25   # Broadcast Overlay

    # --- Draw boxes ---

    # 1. Broadcast Video Input
    draw_box(ax, cx, y1, full_w, box_h,
             "Broadcast Video Input",
             "Raw 1080p60 Match Footage",
             '#F5F5F5', '#757575')

    # 2. Minimap Detection (left)
    draw_box(ax, cx_left, y2, half_w, box_h,
             "Minimap Detection",
             "YOLOv8 Player Counts",
             '#FFF3E0', '#F57C00')

    # 3. Scoreboard OCR (right)
    draw_box(ax, cx_right, y2, half_w, box_h,
             "Scoreboard OCR",
             "Template Matching",
             '#E3F2FD', '#1976D2')

    # 4. Feature Engineering
    draw_box(ax, cx, y3, full_w, box_h,
             "Feature Engineering",
             "12 Features (6 Score + 6 Minimap)",
             '#E8F5E9', '#388E3C')

    # 5. XGBoost Classifier
    draw_box(ax, cx, y4, full_w, box_h,
             "XGBoost Classifier",
             "Win Probability Prediction",
             '#F3E5F5', '#7B1FA2')

    # 6. Broadcast Overlay
    draw_box(ax, cx, y5, full_w, box_h,
             "Broadcast Overlay",
             "Real-Time Visualization",
             '#F5F5F5', '#757575')

    # --- Draw arrows ---
    gap = 0.08

    # Box 1 -> Box 2 (diverge left)
    draw_arrow(ax, cx - 0.35, y1 - box_h / 2 - gap,
               cx_left, y2 + box_h / 2 + gap)

    # Box 1 -> Box 3 (diverge right)
    draw_arrow(ax, cx + 0.35, y1 - box_h / 2 - gap,
               cx_right, y2 + box_h / 2 + gap)

    # Box 2 -> Box 4 (converge left)
    draw_arrow(ax, cx_left, y2 - box_h / 2 - gap,
               cx - 0.35, y3 + box_h / 2 + gap)

    # Box 3 -> Box 4 (converge right)
    draw_arrow(ax, cx_right, y2 - box_h / 2 - gap,
               cx + 0.35, y3 + box_h / 2 + gap)

    # Box 4 -> Box 5
    draw_arrow(ax, cx, y3 - box_h / 2 - gap,
               cx, y4 + box_h / 2 + gap)

    # Box 5 -> Box 6
    draw_arrow(ax, cx, y4 - box_h / 2 - gap,
               cx, y5 + box_h / 2 + gap)

    # --- Save ---
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    plt.savefig(OUTPUT, dpi=300, bbox_inches='tight',
                pad_inches=0.1, facecolor='white')
    plt.close()

    # Also copy to root for README
    shutil.copy2(OUTPUT, ROOT_COPY)

    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"Pipeline diagram saved -> {OUTPUT} ({size_kb:.0f} KB)")
    print(f"Copy saved -> {ROOT_COPY}")


if __name__ == "__main__":
    main()
