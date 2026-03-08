#!/usr/bin/env python3
"""
Organize labeled data into YOLO training dataset structure.

Creates:
- dataset/images/train and dataset/images/val
- dataset/labels/train and dataset/labels/val
- dataset.yaml configuration file
"""

import json
import random
import shutil
from pathlib import Path

# Configuration
FRAMES_DIR = Path("data/frames")
DATASET_DIR = Path("dataset")
CLASSES = ["player", "enemy"]
TRAIN_RATIO = 0.8  # 80% train, 20% val
RANDOM_SEED = 42  # For reproducibility


def convert_labelme_to_yolo(json_path):
    """Convert LabelMe JSON to YOLO format lines."""
    with open(json_path, "r") as f:
        data = json.load(f)

    img_width = data.get("imageWidth", 640)
    img_height = data.get("imageHeight", 640)

    yolo_lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue

        label = shape.get("label", "").lower()
        if label not in CLASSES:
            continue

        class_id = CLASSES.index(label)
        points = shape.get("points", [])

        if len(points) != 2:
            continue

        # LabelMe rectangle: [[x1, y1], [x2, y2]]
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Ensure correct order
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Convert to YOLO format (normalized x_center, y_center, width, height)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Clamp values to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines


def main():
    print("=" * 60)
    print("  ORGANIZE DATASET FOR YOLO TRAINING")
    print("=" * 60)

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Create directory structure
    train_images = DATASET_DIR / "images" / "train"
    val_images = DATASET_DIR / "images" / "val"
    train_labels = DATASET_DIR / "labels" / "train"
    val_labels = DATASET_DIR / "labels" / "val"

    for dir_path in [train_images, val_images, train_labels, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Clean existing files
        for f in dir_path.glob("*"):
            f.unlink()

    print(f"\nCreated directory structure:")
    print(f"  {DATASET_DIR}/")
    print(f"    images/train/")
    print(f"    images/val/")
    print(f"    labels/train/")
    print(f"    labels/val/")

    # Find all JSON files with matching images
    json_files = list(FRAMES_DIR.glob("*.json"))
    print(f"\nFound {len(json_files)} JSON annotation files")

    train_count = 0
    val_count = 0
    skipped_count = 0

    for json_path in json_files:
        img_name = json_path.stem + ".jpg"
        img_path = FRAMES_DIR / img_name

        if not img_path.exists():
            skipped_count += 1
            continue

        # Convert to YOLO format
        yolo_lines = convert_labelme_to_yolo(json_path)

        if not yolo_lines:
            skipped_count += 1
            continue

        # Randomly assign to train or val
        is_train = random.random() < TRAIN_RATIO

        if is_train:
            dest_img = train_images / img_name
            dest_label = train_labels / (json_path.stem + ".txt")
            train_count += 1
        else:
            dest_img = val_images / img_name
            dest_label = val_labels / (json_path.stem + ".txt")
            val_count += 1

        # Copy image
        shutil.copy(img_path, dest_img)

        # Write YOLO label file
        with open(dest_label, "w") as f:
            f.write("\n".join(yolo_lines))

    # Create dataset.yaml
    dataset_yaml_content = f"""path: {DATASET_DIR.absolute()}
train: images/train
val: images/val
names:
  0: player
  1: enemy
"""

    yaml_path = Path("dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(dataset_yaml_content)

    # Print summary
    print("\n" + "=" * 60)
    print("  DATASET ORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"\n  Train images: {train_count}")
    print(f"  Val images:   {val_count}")
    print(f"  Skipped:      {skipped_count}")
    print(f"  Total:        {train_count + val_count}")
    print(f"\n  Train/Val split: {train_count/(train_count+val_count)*100:.1f}% / {val_count/(train_count+val_count)*100:.1f}%")
    print(f"\n  Dataset config: {yaml_path.absolute()}")


if __name__ == "__main__":
    main()
