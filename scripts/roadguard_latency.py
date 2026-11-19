
from pathlib import Path
from statistics import mean, median, stdev
import csv
import json
import time

import torch
from ultralytics import YOLO

MODEL_PATH = Path(
    "runs/detect/runs/roadguard/"
    "yolo_baseline/weights/best.pt"
)

IMAGE_DIR = Path(
    "data/raw/RDD2022ES/combined_annotatedv2"
)

OUTPUT_DIR = Path("experiments/latency")

IMAGE_SIZE = 640
WARMUP_RUNS = 3
MEASURED_RUNS = 20


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def percentile(values, percent):
    values = sorted(values)

    position = (len(values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def main():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise RuntimeError("No road images found")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO(str(MODEL_PATH))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("===== ROADGUARD LATENCY TEST =====")
    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Image size:", IMAGE_SIZE)

    # Warm up the model before timing.
    for i in range(WARMUP_RUNS):
        model.predict(
            source=str(images[i % len(images)]),
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )
        synchronize(device)

    latencies = []
    rows = []

    for i in range(MEASURED_RUNS):

        image = images[i % len(images)]

        synchronize(device)
        start = time.perf_counter()

        result = model.predict(
            source=str(image),
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False,
            conf=0.20
        )

        synchronize(device)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

        rows.append({
            "image": image.name,
            "latency_ms": round(elapsed_ms, 3),
            "detections": len(result[0].boxes)
        })

        print(
            f"Image {i + 1}/{MEASURED_RUNS} | "
            f"{elapsed_ms:.2f} ms"
        )

    summary = {
        "model": str(MODEL_PATH),
        "device": device,
        "image_size": IMAGE_SIZE,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "mean_latency_ms": round(mean(latencies), 3),
        "median_latency_ms": round(median(latencies), 3),
        "p95_latency_ms": round(
            percentile(latencies, 95), 3
        ),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "std_latency_ms": round(
            stdev(latencies), 3
        ),
        "approx_fps": round(
            1000 / mean(latencies), 2
        ),
        "limitations": (
            "Local Python inference benchmark. "
            "Includes image loading and preprocessing "
            "but excludes Java GUI startup and "
            "report-writing overhead. Not a "
            "production deployment benchmark."
        )
    }

    with open(
        OUTPUT_DIR / "latency_results.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=4)

    with open(
        OUTPUT_DIR / "latency_details.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "latency_ms",
                "detections"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n===== LATENCY RESULTS =====")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nLATENCY EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
