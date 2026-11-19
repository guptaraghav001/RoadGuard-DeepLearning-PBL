
from ultralytics import YOLO
from pathlib import Path
import torch

DATA = Path("data/yolo_dataset/data.yaml")

if not DATA.is_file():
    raise FileNotFoundError(
        "YOLO dataset not prepared. Run prepare_yolo_dataset.py first."
    )

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("===== ROADGUARD YOLO TRAINING =====")
print("Training device:", device)

model = YOLO("yolo11n.pt")

results = model.train(
    data=str(DATA),
    epochs=5,
    imgsz=640,
    batch=8,
    device=device,
    workers=2,
    project="runs/roadguard",
    name="yolo_baseline",
    exist_ok=True,
    plots=True
)

print("Training completed.")
print("Best model: runs/roadguard/yolo_baseline/weights/best.pt")
