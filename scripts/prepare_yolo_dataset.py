
from pathlib import Path
import csv
import os

ROOT = Path("data")
IMAGE_DIR = ROOT / "raw/RDD2022ES/combined_annotatedv2"
LABEL_DIR = ROOT / "processed/RDD2022ES/labels"
SPLIT_DIR = ROOT / "splits"
OUTPUT = ROOT / "yolo_dataset"

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

for split in ["train", "val", "test"]:
    image_out = OUTPUT / "images" / split
    label_out = OUTPUT / "labels" / split

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    count = 0
    missing = 0

    with open(SPLIT_DIR / f"{split}.csv", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_src = IMAGE_DIR / row["image"]
            label_src = LABEL_DIR / row["label"]

            if not image_src.is_file() or not label_src.is_file():
                missing += 1
                continue

            image_dst = image_out / image_src.name
            label_dst = label_out / (
                image_src.stem + ".txt"
            )

            if not image_dst.exists():
                os.symlink(
                    image_src.resolve(),
                    image_dst
                )

            if not label_dst.exists():
                os.symlink(
                    label_src.resolve(),
                    label_dst
                )

            count += 1

    print(
        f"{split}: {count} linked images, "
        f"{missing} missing pairs"
    )

yaml_path = OUTPUT / "data.yaml"

with open(yaml_path, "w") as f:
    f.write(f"path: {OUTPUT.resolve()}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("nc: 8\n")
    f.write("names:\n")

    for i, name in enumerate(CLASSES):
        f.write(f"  {i}: '{name}'\n")

print("YOLO DATASET PREPARATION COMPLETED")
print("Configuration:", yaml_path)
