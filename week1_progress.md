# Week 1 Progress Report: Esports Computer Vision Pipeline
**Student:** Trevor Gregory
**Project:** Real-Time Win Probability Estimation in Call of Duty (Practicum)
**Date:** January 28, 2026

## 1. Executive Summary
In the first week of development, we successfully established the end-to-end data engineering pipeline. The primary focus was **Data Acquisition** and **Automated Annotation**. To overcome the high labor cost of manual labeling, we implemented a **Model-Assisted Labeling (Active Learning)** workflow, allowing us to generate a dataset of ~700 annotated frames while only manually labeling 50.

## 2. Technical Architecture & Setup

### Hardware Acceleration (Apple Silicon)
* **Challenge:** Deep Learning frameworks traditionally rely on NVIDIA CUDA (GPUs), which is incompatible with Mac architecture.
* **Solution:** We configured a **PyTorch** environment utilizing **MPS (Metal Performance Shaders)**. This allows the project to leverage the Mac’s Neural Engine/GPU for acceleration, reducing training times from hours (CPU) to minutes.

### Data Acquisition
* **Source:** Official Call of Duty League (CDL) broadcasts via YouTube.
* **Tooling:** Utilized `yt-dlp` for high-fidelity extraction of 1080p/60fps match footage.
* **Sampling Strategy:** To prevent overfitting on identical frames, we implemented a **temporal sampling rate of 0.2 FPS** (1 frame every 5 seconds). This ensures the dataset captures a diverse range of game states (rotations, engagements, idle times) rather than repetitive sequences.

## 3. The Computer Vision Pipeline

### Step A: Region of Interest (ROI) Extraction
Instead of processing the entire 1920x1080 frame (which introduces noise from player perspectives and HUD elements), we developed a preprocessing script to perform a fixed-coordinate crop of the **Minimap**.
* **Technique:** Static array slicing on the input tensor.
* **Output:** A clean, square dataset of map-only imagery resized to 640x640 (native YOLOv8 resolution).

### Step B: The Annotation Strategy (Active Learning)
Manually drawing bounding boxes on 700+ images is inefficient. We adopted a **Human-in-the-Loop** approach:

1.  **Cold Start (Manual Labeling):**
    * We manually labeled a "seed set" of **50 images** using `labelme`.
    * Classes defined: `player` (white arrows), `enemy` (red diamonds).
2.  **Shadow Model Training:**
    * We trained a lightweight **YOLOv8-Nano** model on this small seed set for 30 epochs.
    * *Result:* A "weak learner" capable of detecting obvious patterns but prone to errors.
3.  **Inference & Auto-Labeling:**
    * We ran this model against the remaining **688 unlabeled images**.
    * Predictions with a confidence score > 0.25 were converted into LabelMe-compatible JSON format.
4.  **Human Correction:**
    * Instead of drawing boxes from scratch, the annotator (me) simply validates the AI's pre-drawn boxes, correcting false positives or missed detections. This reduced annotation time by approximately **90%**.

## 4. Tools & Technologies Used
* **Language:** Python 3.10
* **Frameworks:** PyTorch (MPS Backend), Ultralytics YOLOv8
* **Data Engineering:** `OpenCV` (Image processing), `yt-dlp` (Scraping), `pandas` (Metadata)
* **Annotation:** `labelme` (JSON-based polygon/rectangle tool)

## 5. Next Steps (Week 2)
* **Finalize Dataset:** Complete the verification of the auto-labeled images.
* **Full Training:** Train a larger model (YOLOv8-Medium) on the complete, clean dataset.
* **OCR Integration:** Begin development of the Tesseract/EasyOCR pipeline to read the game score and clock.