# YOLOv11 Deep Learning Workflow for Cold-Water Coral Assessment (Dendrophyllia cornigera)

This repository contains the full source code and analysis workflows developed for the paper **"A Comparative Assessment of a Deep Learning Algorithm and Human Experts in the Quantification of Cold-Water Corals (*Dendrophyllia cornigera*)"**.

The project implements a modular Deep Learning pipeline based on the YOLOv11-L architecture to automate the counting and density estimation of *D. cornigera* in Vulnerable Marine Ecosystems (VMEs) from ROV video data. The core novelty is the use of a segmentation module to dynamically correct the Effective Analysis Area (AAEm2), minimizing geometric sampling bias.

## 🚀 Key Features

* **YOLOv11-L Implementation:** Three specialized models for Classification, Detection, and Segmentation.
* **Geometric Bias Correction:** Novel segmentation module to define the AAEm2.
* **Comparative Analysis:** Code for replicating the **Segment Method** (corrected area) and the **Rectangle Method** (fixed area simulation).
* **Hyperparameter Optimization:** Includes the Optuna script used to find the optimal training configuration.
* [cite_start]**Robustness Evaluation:** K-Fold Cross-Validation scripts for model robustness assessment[cite: 126].

## 🛠️ Setup and Installation

### Prerequisites

You need a machine with a dedicated **NVIDIA GPU** for efficient training and inference. [cite_start]The code was developed using an **NVIDIA GeForce RTX 4070**[cite: 98].

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/AI-for-VME-Assessment.git](https://github.com/your-username/AI-for-VME-Assessment.git)
    cd AI-for-VME-Assessment
    ```
2.  **Install Dependencies:**
    The core dependencies are `ultralytics` (for YOLOv11), `torch`, `opencv`, `pandas`, and `optuna`.
    ```bash
    pip install -r requirements.txt
    ```
    *(You need to create the `requirements.txt` file listing all libraries used in your scripts: `ultralytics`, `torch`, `torchvision`, `optuna`, `pandas`, `opencv-python`, `scikit-learn`, `PyYAML`).*

### Data and Weights

The original dataset is hosted externally due to size constraints.
* **Data Structure:** The `datasets/` directory should contain the input data structured as defined in `label1.yaml` (including images and YOLO-format labels).
* **Pre-trained Weights:** Download the pre-trained weights (`detection.pt`, `segment.pt`, `classify.pt`) from [Link to your external storage/Zenodo/etc.] and place them in the `weights/` directory.

## ⚙️ Usage: Reproducing the Results

### 1. Training and Optimization

The final models were trained using optimized hyperparameters.

| Script | Purpose |
| :--- | :--- |
| `training_scripts/optimize_yolo.py` | Runs the Optuna search (50 trials) to maximize the detection mAP50-95. |
| `training_scripts/train-det.py` | Final training script for the Detection Model using the optimal hyperparameters. |
| `training_scripts/kfold_validation/kfold-det.py` | [cite_start]Reproduces the K-Fold Cross-Validation for the Detection Model[cite: 126]. |

### 2. Analysis Workflows (Density Estimation)

These scripts replicate the core comparison against the "Gold Standard" protocol. Both scripts require a **telemetry CSV** file and the **ROV video** as input.

| Script | Protocol Replicated | Ecological Metric |
| :--- | :--- | :--- |
| `analysis_workflows/rectangle_method.py` | [cite_start]**Scenario 1:** Fixed-Area Simulation (The "Gold Standard" protocol [cite: 145]) | Density (n/fixed $m^2$) |
| `analysis_workflows/segment_method.py` | [cite_start]**Scenario 2:** Full YOLOv11 Workflow (Methodologically Superior [cite: 208]) | Corrected Density (n/AAEm2) |

**Example Execution:**
```bash
python analysis_workflows/segment_method.py
# (The script will prompt you for the telemetry file name and video path)
