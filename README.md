# RoadGuard — Deep Learning PBL

**Urban Pothole and Road Damage Assessment System Using Deep Learning**

An individual Deep Learning Project-Based Learning (PBL) project developed as part of the Semester V curriculum at UPES.

## 1. Project Overview

RoadGuard is a deep learning-based road damage assessment system designed to detect and classify road surface defects from photographs.

The project combines a custom Convolutional Neural Network (CNN), a fine-tuned YOLO object detector and a Java Swing desktop application.

The system allows users to select road photographs, identify potential road damage, visualize predicted bounding boxes and generate structured assessment reports.

Additional experiments demonstrate LSTM-based sequence prediction, VAE-based image generation and bounding-box mask visualization.

## 2. Project Objectives

- Detect and localize road surface damage in photographs.
- Classify damage into eight categories.
- Train and evaluate a CNN using road-damage image crops.
- Fine-tune a YOLO detector for full-image object detection.
- Generate annotated images and JSON assessment reports.
- Provide a desktop application for interactive inference.
- Investigate LSTM and VAE architectures through supporting experiments.
- Evaluate model performance using appropriate metrics.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Primary application | Java, Java Swing |
| Java build system | Apache Maven |
| Java deep learning | Deep Java Library (DJL) |
| Object detection | Ultralytics YOLO11 |
| Python deep learning | PyTorch |
| Image processing | OpenCV, Pillow |
| Data processing | NumPy |
| Experiment configuration | YAML, JSON |
| Development environment | Visual Studio Code |
| Operating system used | macOS, Apple Silicon |
| Version control | Git and GitHub |

The Java project uses Java 17-compatible Maven compilation. The Python detection experiments were developed using Python 3.12.

## 4. Dataset

**Dataset:** RDD2022ES — Road Damage Detection Extended Severity Dataset.

The project uses road photographs with bounding-box annotations representing eight road-damage categories.

### Damage categories

| Class ID | Category |
|---|---|
| 0 | Longitudinal Crack |
| 1 | Longitudinal Crack Wide |
| 2 | Transverse Crack |
| 3 | Transverse Crack Wide |
| 4 | Alligator Crack |
| 5 | Alligator Crack Sunken |
| 6 | Pothole |
| 7 | Pothole Deep |

### Dataset preparation

The dataset preparation pipeline includes:

- Image and annotation inspection.
- Cleaning and validation of bounding-box annotations.
- Train, validation and test split generation.
- Duplicate and leakage checks.
- Generation of CNN classification crops.
- YOLO detection dataset preparation.

The processed CNN training dataset contains **28,632 damage crops**.

The final test split used for CNN evaluation contains **1,832 damage crops**.

The complete original dataset and generated image files are not included in this GitHub repository because of their size.

## 5. Project Architecture

RoadGuard contains two main computer vision pipelines.

### CNN classification pipeline

Road image → Annotated damage region → Crop preprocessing → CNN → Damage class prediction

The CNN was implemented using Java and DJL with the PyTorch backend.

### YOLO detection pipeline

Road photograph → YOLO detector → Bounding-box predictions → Damage categories and confidence scores → Assessment report

The YOLO model was fine-tuned using the prepared road-damage detection dataset.

### Desktop application

The Java Swing application allows users to:

1. Select a road photograph.
2. Run the trained YOLO detector.
3. Display an annotated image.
4. Review detected damage categories and confidence scores.
5. View the estimated maintenance priority.
6. Open the generated JSON assessment report.

The maintenance-priority assignment uses predefined class-based rules. It is an illustrative heuristic, not a validated engineering severity assessment.

## 6. CNN Training and Evaluation

A custom CNN classifier was trained on road-damage crops using DJL.

### Training configuration

| Parameter | Value |
|---|---|
| Input resolution | 224 × 224 |
| Number of classes | 8 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Softmax cross-entropy |
| Training epochs | 2 |

### CNN baseline results

| Metric | Result |
|---|---:|
| Test images | 1,832 |
| Correct predictions | 1,044 |
| Test accuracy | 56.99% |
| Macro-F1 | 0.2518 |

The baseline performed better on some frequently represented damage classes than on minority classes.

Class imbalance and visually similar damage categories were identified as important limitations.

An improved CNN Version 2 architecture was also developed and tested with a forward pass. The baseline results above do not represent a completed Version 2 training experiment.

## 7. YOLO Object Detection

RoadGuard uses a fine-tuned YOLO11 nano model for full-image road-damage detection.

### Training configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Input image size | 640 |
| Epochs | 5 |
| Batch size | 8 |
| Training acceleration | Apple Metal Performance Shaders (MPS) |
| Detection classes | 8 |

### YOLO test results

| Metric | Result |
|---|---:|
| Precision | 0.3811 |
| Recall | 0.3943 |
| mAP@50 | 0.3309 |
| mAP@50–95 | 0.1633 |

These metrics were obtained from the held-out test split.

The results demonstrate a functioning detector but also show that localization and classification performance require improvement before real-world deployment.

## 8. Additional Deep Learning Experiments

### Bounding-box mask visualization

A Python script generates binary masks and colored overlays using predicted YOLO bounding boxes.

This is a rectangular mask approximation, not a trained semantic or instance segmentation model.

The reported coverage represents bounding-box area rather than confirmed damaged road surface.

### LSTM deterioration demonstration

A small LSTM model was trained using synthetic road-condition sequences.

| Parameter | Value |
|---|---|
| Sequence length | 6 |
| Training samples | 480 |
| Test samples | 120 |
| Epochs | 30 |
| MAE | 0.097775 |
| RMSE | 0.116905 |

This experiment demonstrates sequence learning. It is not validated for real-world road deterioration forecasting.

### Variational Autoencoder

A small VAE was trained using road-damage crops to demonstrate latent-space image generation.

| Parameter | Value |
|---|---|
| Training images | 2,000 |
| Epochs | 5 |
| Latent dimension | 32 |

The experiment saves generated samples and model parameters.

The generated images have not been validated as realistic or label-preserving training augmentation.

## 9. Project Structure

```text
road-damage-pbl/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── yolo_dataset/
├── experiments/
├── models/
├── outputs/
├── runs/
├── scripts/
│   ├── prepare_yolo_dataset.py
│   ├── train_yolo.py
│   ├── evaluate_yolo.py
│   ├── roadguard_inference.py
│   ├── roadguard_segmentation.py
│   ├── roadguard_lstm.py
│   └── roadguard_vae.py
├── src/
│   ├── main/java/com/roadguard/
│   └── test/
├── pom.xml
├── .gitignore
└── README.md
```

Some generated directories are intentionally excluded from GitHub.

## 10. Installation and Setup

### Requirements

- Java Development Kit compatible with Java 17 compilation.
- Apache Maven.
- Python 3.12.
- Python packages required by the detection and supporting experiments.
- Locally available trained model checkpoints.
- RDD2022ES dataset or suitable sample road images.

### Java project

Compile:

```bash
mvn clean compile
```

Run unit tests:

```bash
mvn clean test
```

The verified Java test suite completed 35 tests with zero failures and errors.

### Python environment

Create a virtual environment:

```bash
python3.12 -m venv .venv-yolo
```

Activate it:

```bash
source .venv-yolo/bin/activate
```

Install dependencies:

```bash
python -m pip install ultralytics torch torchvision opencv-python numpy pillow pyyaml
```

For exact reproduction, package versions should be recorded from the tested environment.

## 11. Running RoadGuard

### Launch the desktop application

Ensure the Maven execution plugin is configured to launch:

`com.roadguard.RoadGuardGUI`

Run:

```bash
mvn exec:java
```

The Java Swing desktop window will open.

Select a road photograph and click **Analyze Road**.

### Command-line inference

The existing Java command-line application can be launched by explicitly selecting its main class:

```bash
mvn exec:java -Dexec.mainClass=com.roadguard.RoadGuardApplication -Dexec.args="path/to/road_image.jpg"
```

### Python inference

```bash
python scripts/roadguard_inference.py --image path/to/road_image.jpg
```

The trained YOLO checkpoint is expected at:

`runs/detect/runs/roadguard/yolo_baseline/weights/best.pt`

### Generated outputs

The application saves annotated photographs and JSON reports inside:

`outputs/roadguard/`

## 12. Model Checkpoints

The project uses saved checkpoints for inference and experimental reproducibility.

| Model | Checkpoint location |
|---|---|
| CNN baseline | `models/roadguard_cnn/` |
| YOLO detector | `runs/detect/runs/roadguard/yolo_baseline/weights/best.pt` |
| LSTM | `experiments/lstm/roadguard_lstm.pt` |
| VAE | `experiments/vae/roadguard_vae.pt` |

These files are excluded from the public GitHub repository.

To reproduce inference, the checkpoints must be restored locally or regenerated using the relevant training scripts.

## 13. Limitations

The current system is an academic prototype.

Known limitations include:

- Moderate detection accuracy and imperfect bounding-box localization.
- Class imbalance in the classification dataset.
- Potential false-positive and missed detections.
- Heuristic maintenance-priority assignment.
- Bounding-box masks rather than true pixel-level segmentation.
- Synthetic rather than real longitudinal data for the LSTM.
- Unvalidated realism of VAE-generated images.
- No validated geolocation or GPS error measurements.

RoadGuard should not be used as the sole basis for road-maintenance or safety decisions.

## 14. Future Improvements

Potential improvements include:

- Longer and better-tuned YOLO training.
- Class-balanced sampling and data augmentation.
- Stronger CNN architectures and transfer learning.
- True semantic or instance segmentation.
- Road-condition tracking with real temporal observations.
- GPS-enabled damage reporting.
- Deployment as a web or mobile application.
- Validation on diverse real-world road conditions.

## 15. Academic Integrity and Acknowledgements

RoadGuard was developed as an individual university PBL project.

The project uses the RDD2022ES dataset, open-source deep learning libraries and pretrained YOLO weights.

ChatGPT was used as an AI-assisted development and learning tool for implementation guidance, debugging, code explanations and documentation.

The pretrained components, supporting libraries and AI assistance should be acknowledged when presenting the project.

## 16. Evaluation

The project is prepared for the Semester V Deep Learning individual evaluation scheduled for **23 November 2026**.

The demonstration focuses on the working RoadGuard application, dataset processing, model architecture, training methodology, experimental results and limitations.

The technical viva covers mathematical understanding, implementation decisions, evaluation metrics and interpretation of model errors.

---

**RoadGuard — Urban Pothole and Road Damage Assessment System**

*Deep Learning PBL | Semester V | UPES*