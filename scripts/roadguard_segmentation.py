
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==========================================
# ROADGUARD CONFIGURATION
# ==========================================

MODEL_PATH = Path(
    "runs/detect/runs/roadguard/"
    "yolo_baseline/weights/best.pt"
)

OUTPUT_DIR = Path("outputs/segmentation")


# ==========================================
# MAIN SEGMENTATION FUNCTION
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="RoadGuard Damage Segmentation Demo"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to road image"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="YOLO confidence threshold"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    # Validate input image
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Validate trained YOLO model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    # Read original image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    height, width = image.shape[:2]

    print("===== ROADGUARD SEGMENTATION =====")
    print("Input image:", image_path)
    print("Image dimensions:", width, "x", height)

    # ==========================================
    # LOAD TRAINED YOLO MODEL
    # ==========================================

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Inference device:", device)

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        imgsz=640,
        device=device,
        verbose=False
    )

    result = results[0]

    # ==========================================
    # GENERATE BINARY MASK
    # ==========================================

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    detections = []

    for box in result.boxes:

        # Convert coordinates to standard Python integers
        x1, y1, x2, y2 = [
            int(v)
            for v in box.xyxy[0].cpu().tolist()
        ]

        # Keep coordinates within image boundaries
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))

        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        # Skip invalid bounding boxes
        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        # Fill detected bounding-box region
        mask[y1:y2, x1:x2] = 255

        detections.append({
            "class_id": class_id,
            "class_name": str(
                model.names[class_id]
            ),
            "confidence": round(
                confidence, 4
            ),
            "bounding_box": [
                x1, y1, x2, y2
            ]
        })

    # ==========================================
    # GENERATE COLORED OVERLAY
    # ==========================================

    colored = image.copy()

    # Red highlight over detected regions
    colored[mask > 0] = (0, 0, 255)

    overlay = cv2.addWeighted(
        image,
        0.65,
        colored,
        0.35,
        0
    )

    # ==========================================
    # CALCULATE COVERAGE
    # ==========================================

    coverage = (
        np.count_nonzero(mask)
        / mask.size
        * 100.0
    )

    # ==========================================
    # CREATE OUTPUT DIRECTORY AND FILE PATHS
    # ==========================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_path = OUTPUT_DIR / (
        image_path.stem + "_mask.png"
    )

    overlay_path = OUTPUT_DIR / (
        image_path.stem + "_overlay.jpg"
    )

    report_path = OUTPUT_DIR / (
        image_path.stem + "_segmentation.json"
    )

    # ==========================================
    # SAVE IMAGES
    # ==========================================

    if not cv2.imwrite(
        str(mask_path), mask
    ):
        raise IOError(
            "Failed to save binary mask"
        )

    if not cv2.imwrite(
        str(overlay_path), overlay
    ):
        raise IOError(
            "Failed to save overlay image"
        )

    # ==========================================
    # CREATE JSON REPORT
    # ==========================================

    report = {
        "project": "RoadGuard",
        "input_image": str(image_path),
        "model": str(MODEL_PATH),
        "method": (
            "YOLO bounding-box mask approximation"
        ),
        "device": device,
        "image_width": int(width),
        "image_height": int(height),
        "total_detections": len(detections),
        "detections": detections,
        "bounding_box_coverage_percent": round(
            float(coverage), 2
        ),
        "mask_file": str(mask_path),
        "overlay_file": str(overlay_path),
        "limitations": (
            "This is a rectangular bounding-box "
            "approximation, not pixel-level "
            "semantic segmentation. Coverage "
            "represents predicted box area rather "
            "than confirmed damaged road area."
        )
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ==========================================
    # FINAL OUTPUT
    # ==========================================

    print("\n===== SEGMENTATION RESULTS =====")

    print(
        "Detections:",
        len(detections)
    )

    print(
        "Bounding-box coverage:",
        round(float(coverage), 2),
        "%"
    )

    print("Binary mask:", mask_path)
    print("Overlay image:", overlay_path)
    print("JSON report:", report_path)

    print("\nSEGMENTATION DEMO COMPLETED")


if __name__ == "__main__":
    main()
