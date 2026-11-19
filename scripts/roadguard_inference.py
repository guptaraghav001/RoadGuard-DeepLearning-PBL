
from ultralytics import YOLO
from pathlib import Path
import argparse
import json
import torch

MODEL = Path(
    "runs/detect/runs/roadguard/"
    "yolo_baseline/weights/best.pt"
)

OUTPUT = Path("outputs/roadguard")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Longitudinal Crack",
    "Longitudinal Crack Wide",
    "Transverse Crack",
    "Transverse Crack Wide",
    "Alligator Crack",
    "Alligator Crack Sunken",
    "Pothole",
    "Pothole Deep"
]

def priority_for_class(class_id):
    if class_id in (5, 7):
        return "HIGH"
    if class_id in (1, 3, 4, 6):
        return "MEDIUM"
    return "LOW"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf", type=float, default=0.20)
    args = parser.parse_args()

    image = Path(args.image)

    if not image.is_file():
        raise FileNotFoundError(image)

    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO(str(MODEL))

    results = model.predict(
        source=str(image),
        conf=args.conf,
        imgsz=640,
        device=device,
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        coordinates = [
            round(float(v), 2)
            for v in box.xyxy[0].tolist()
        ]

        detections.append({
            "class_id": class_id,
            "class_name": CLASSES[class_id],
            "confidence": round(confidence, 4),
            "bounding_box": coordinates,
            "priority": priority_for_class(class_id)
        })

    annotated = result.plot()

    import cv2

    output_image = OUTPUT / (
        image.stem + "_annotated.jpg"
    )

    cv2.imwrite(
        str(output_image),
        annotated
    )

    priorities = [
        d["priority"] for d in detections
    ]

    if "HIGH" in priorities:
        overall_priority = "HIGH"
    elif "MEDIUM" in priorities:
        overall_priority = "MEDIUM"
    elif "LOW" in priorities:
        overall_priority = "LOW"
    else:
        overall_priority = "NO DAMAGE DETECTED"

    report = {
        "input_image": str(image),
        "model": str(MODEL),
        "detections": detections,
        "total_detections": len(detections),
        "overall_priority": overall_priority,
        "annotated_image": str(output_image)
    }

    report_path = OUTPUT / (
        image.stem + "_report.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print("===== ROADGUARD ASSESSMENT =====")
    print("Image:", image)
    print("Detections:", len(detections))
    print("Maintenance priority:", overall_priority)
    print("Annotated image:", output_image)
    print("Report:", report_path)

if __name__ == "__main__":
    main()
