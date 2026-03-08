#!/usr/bin/env python3
"""
Train production YOLOv8 model for Call of Duty minimap detection.
"""

from ultralytics import YOLO


def main():
    print("=" * 60)
    print("  TRAINING PRODUCTION MODEL")
    print("  YOLOv8 Medium - COD Minimap Detection")
    print("=" * 60)

    # Load YOLOv8 Medium pretrained model
    print("\nLoading yolov8m.pt (Medium) pretrained model...")
    model = YOLO("yolov8m.pt")

    # Train the model
    print("\nStarting training...")
    print("  Epochs: 100")
    print("  Image size: 640")
    print("  Batch size: 16")
    print("  Device: MPS (Apple Silicon GPU)")
    print()

    results = model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="mps",
        project="runs/detect",
        name="cod_model_v1",
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModel saved to: runs/detect/cod_model_v1/weights/best.pt")
    print("\nTo use the model:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('runs/detect/cod_model_v1/weights/best.pt')")
    print("  results = model.predict('your_image.jpg')")


if __name__ == "__main__":
    main()
