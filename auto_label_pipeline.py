#!/usr/bin/env python3
"""
Model-Assisted Labeling Pipeline for Esports CV

Step 1: Convert existing LabelMe JSON annotations to YOLO format
Step 2: Train a 'baby' YOLOv8 model on the labeled data
Step 3: Run inference on unlabeled images
Step 4: Convert YOLO predictions back to LabelMe JSON format
"""

import json
import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# Configuration
FRAMES_DIR = Path("data/frames")
DATASET_DIR = Path("data/yolo_dataset")
CLASSES = ["player", "enemy"]
IMG_SIZE = 640
EPOCHS = 30
CONFIDENCE_THRESHOLD = 0.25


def step1_convert_labelme_to_yolo():
    """Convert existing LabelMe JSON annotations to YOLO format."""
    print("=" * 60)
    print("STEP 1: Converting LabelMe JSON to YOLO format")
    print("=" * 60)

    # Create YOLO dataset structure
    train_images = DATASET_DIR / "images" / "train"
    train_labels = DATASET_DIR / "labels" / "train"
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(FRAMES_DIR.glob("*.json"))
    print(f"Found {len(json_files)} LabelMe JSON files")

    if len(json_files) == 0:
        print("ERROR: No JSON files found. Please label some images first.")
        return False

    converted_count = 0

    for json_path in json_files:
        # Find corresponding image
        img_name = json_path.stem + ".jpg"
        img_path = FRAMES_DIR / img_name

        if not img_path.exists():
            print(f"  Warning: Image not found for {json_path.name}, skipping")
            continue

        # Load LabelMe JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        img_width = data.get("imageWidth", IMG_SIZE)
        img_height = data.get("imageHeight", IMG_SIZE)

        # Convert annotations
        yolo_lines = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue

            label = shape.get("label", "").lower()
            if label not in CLASSES:
                print(f"  Warning: Unknown class '{label}' in {json_path.name}, skipping")
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

        if yolo_lines:
            # Copy image to dataset
            shutil.copy(img_path, train_images / img_name)

            # Write YOLO label file
            label_path = train_labels / (json_path.stem + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

            converted_count += 1

    print(f"Converted {converted_count} images with annotations")

    # Create dataset.yaml
    dataset_yaml = {
        "path": str(DATASET_DIR.absolute()),
        "train": "images/train",
        "val": "images/train",  # Using same for simplicity with small dataset
        "names": {i: name for i, name in enumerate(CLASSES)}
    }

    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"Created dataset.yaml at {yaml_path}")
    return True


def step2_train_baby_model():
    """Train a YOLOv8 nano model on the labeled data."""
    print("\n" + "=" * 60)
    print("STEP 2: Training 'Baby' YOLOv8 Model")
    print("=" * 60)

    yaml_path = DATASET_DIR / "dataset.yaml"

    if not yaml_path.exists():
        print("ERROR: dataset.yaml not found. Run Step 1 first.")
        return None

    # Load pretrained YOLOv8 nano
    print("Loading yolov8n.pt (nano) pretrained model...")
    model = YOLO("yolov8n.pt")

    # Train the model
    print(f"Training for {EPOCHS} epochs on MPS device...")
    print(f"Image size: {IMG_SIZE}")

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device="mps",
        project="runs",
        name="baby_model",
        exist_ok=True,
        verbose=True
    )

    # Get path to best model - search for it since YOLO may nest directories
    possible_paths = [
        Path("runs/baby_model/weights/best.pt"),
        Path("runs/detect/baby_model/weights/best.pt"),
        Path("runs/detect/runs/baby_model/weights/best.pt"),
    ]

    for best_path in possible_paths:
        if best_path.exists():
            print(f"Training complete! Best model saved at: {best_path}")
            return best_path

    # Fallback: search for any best.pt in runs directory
    for best_path in Path("runs").rglob("best.pt"):
        print(f"Training complete! Best model saved at: {best_path}")
        return best_path

    # Last resort: find last.pt
    for last_path in Path("runs").rglob("last.pt"):
        print(f"Training complete! Model saved at: {last_path}")
        return last_path

    print("ERROR: Could not find trained model weights")
    return None


def step3_predict_unlabeled(model_path):
    """Run inference on images without JSON annotations."""
    print("\n" + "=" * 60)
    print("STEP 3: Predicting on Unlabeled Images")
    print("=" * 60)

    # Find all images
    all_images = set(p.stem for p in FRAMES_DIR.glob("*.jpg"))

    # Find images that already have JSON
    labeled_images = set(p.stem for p in FRAMES_DIR.glob("*.json"))

    # Get unlabeled images
    unlabeled = all_images - labeled_images
    unlabeled_paths = [FRAMES_DIR / f"{name}.jpg" for name in sorted(unlabeled)]

    print(f"Total images: {len(all_images)}")
    print(f"Already labeled: {len(labeled_images)}")
    print(f"To predict: {len(unlabeled_paths)}")

    if len(unlabeled_paths) == 0:
        print("All images are already labeled!")
        return {}

    # Load our trained model
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))

    # Run inference
    print(f"Running inference with confidence threshold: {CONFIDENCE_THRESHOLD}")
    predictions = {}

    results = model.predict(
        source=unlabeled_paths,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMG_SIZE,
        device="mps",
        verbose=False
    )

    for result in results:
        img_path = Path(result.path)
        img_name = img_path.stem

        boxes = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            # Get xyxy coordinates (pixel values)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            boxes.append({
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        predictions[img_name] = {
            "boxes": boxes,
            "img_width": result.orig_shape[1],
            "img_height": result.orig_shape[0]
        }

    total_boxes = sum(len(p["boxes"]) for p in predictions.values())
    print(f"Generated {total_boxes} predictions across {len(predictions)} images")

    return predictions


def step4_generate_labelme_jsons(predictions):
    """Convert YOLO predictions to LabelMe JSON format."""
    print("\n" + "=" * 60)
    print("STEP 4: Generating LabelMe JSON Files")
    print("=" * 60)

    if not predictions:
        print("No predictions to convert.")
        return

    generated_count = 0

    for img_name, pred_data in predictions.items():
        img_path = FRAMES_DIR / f"{img_name}.jpg"
        json_path = FRAMES_DIR / f"{img_name}.json"

        # Skip if JSON already exists (shouldn't happen, but safety check)
        if json_path.exists():
            continue

        # Build LabelMe JSON structure
        shapes = []
        for box in pred_data["boxes"]:
            x1, y1, x2, y2 = box["bbox"]

            shape = {
                "label": box["class_name"],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": f"auto-labeled (conf: {box['confidence']:.2f})",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            shapes.append(shape)

        labelme_data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"{img_name}.jpg",
            "imageData": None,  # LabelMe will auto-load from disk
            "imageHeight": pred_data["img_height"],
            "imageWidth": pred_data["img_width"]
        }

        # Write JSON file
        with open(json_path, "w") as f:
            json.dump(labelme_data, f, indent=2)

        generated_count += 1

    print(f"Generated {generated_count} new LabelMe JSON files")
    print(f"Files saved to: {FRAMES_DIR}")


def main():
    print("\n" + "=" * 60)
    print("  MODEL-ASSISTED LABELING PIPELINE")
    print("  Esports Computer Vision Project")
    print("=" * 60)

    # Step 1: Convert existing labels
    if not step1_convert_labelme_to_yolo():
        print("\nPipeline aborted due to Step 1 failure.")
        return

    # Step 2: Train baby model
    model_path = step2_train_baby_model()
    if model_path is None:
        print("\nPipeline aborted due to Step 2 failure.")
        return

    # Step 3: Predict unlabeled images
    predictions = step3_predict_unlabeled(model_path)

    # Step 4: Generate LabelMe JSONs
    step4_generate_labelme_jsons(predictions)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Open labelme to review and correct the auto-generated labels")
    print("2. Run this script again to improve the model with more data")
    print("3. Repeat until satisfied with label quality")


if __name__ == "__main__":
    main()
