
from pathlib import Path
from ultralytics import YOLO
import torch
import json
import csv

# Project paths
MODEL_PATH = Path(
    "runs/detect/runs/roadguard/"
    "yolo_baseline/weights/best.pt"
)

DATA_PATH = Path("data/yolo_dataset/data.yaml")

OUTPUT_DIR = Path("experiments/yolo_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained YOLO model not found: {MODEL_PATH}"
    )

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset configuration missing: {DATA_PATH}"
    )

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("===== ROADGUARD YOLO EVALUATION =====")
print("Model:", MODEL_PATH)
print("Device:", device)

model = YOLO(str(MODEL_PATH))

# Evaluate on the held-out test split.
metrics = model.val(
    data=str(DATA_PATH),
    split="test",
    imgsz=640,
    batch=8,
    device=device,
    workers=2,
    project=str(OUTPUT_DIR),
    name="test_results",
    exist_ok=True,
    plots=True
)

results = {
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "mAP50": float(metrics.box.map50),
    "mAP50_95": float(metrics.box.map),
    "device": device,
    "image_size": 640,
    "model": str(MODEL_PATH),
}

# Save summary as JSON.
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# Save summary as CSV.
with open(
    OUTPUT_DIR / "metrics.csv",
    "w",
    newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])

    for key, value in results.items():
        writer.writerow([key, value])

print("\n===== FINAL YOLO METRICS =====")

for key, value in results.items():
    print(f"{key}: {value}")

print("\nEvaluation results saved to:", OUTPUT_DIR)
print("ROADGUARD YOLO EVALUATION COMPLETED")
