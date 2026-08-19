# AT-MGAN Intrusion Detection System

This project implements an AT-MGAN-based intrusion detection system using the UNSW-NB15 and IoT Telemetry IDS datasets. It compares six GAN-based variants: GAN, ACGAN, MGAN, MGAN with feature-distribution loss, MGAN with feature-distribution and cosine loss, and the proposed AT-MGAN.

## Project Files

- `01_preprocessing.py` — Preprocesses UNSW-NB15 and IoT Telemetry datasets.
- `02_models.py` — Defines Generator, Discriminator, and loss functions.
- `03_train.py` — Trains all GAN variants and saves model checkpoints.
- `04_evaluate.py` — Evaluates trained models and saves ablation results.
- `05_run_all.py` — Runs preprocessing, training, and evaluation sequentially.
- `ablation-study2-fin.py` — Optimized complete pipeline with SMOTE, early stopping, ROC curves, loss curves, and result plots.
- `processed/` — Stores processed datasets and train/test arrays.
- `checkpoints/` — Stores trained Generator and Discriminator weights.
- `results/` — Stores evaluation tables, reports, curves, and plots.

## Step 1: Prepare Datasets

Place the following files in the project root directory:

```text
UNSW-NB15.csv
iot_telemetry_ids.csv
```

The IoT Telemetry dataset must contain these columns:

```text
co, humidity, light, lpg, motion, smoke, temp, label
```

## Step 2: Install Requirements

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib torch
```

## Step 3: Preprocess Data

Run:

```bash
python 01_preprocessing.py
```

This script performs the following operations:

- Loads UNSW-NB15 and IoT Telemetry datasets.
- Removes unnecessary columns from UNSW-NB15.
- Converts categorical UNSW-NB15 features into numerical values.
- Converts IoT `light` and `motion` Boolean values into binary values.
- Handles missing values.
- Selects relevant features.
- Standardizes feature values.
- Applies PCA.
- Saves processed arrays in the `processed/` directory.

For UNSW-NB15, the pipeline selects 30 features and reduces them to 20 PCA components. For IoT Telemetry, it uses seven features and reduces them to five PCA components. [2]

## Step 4: Train GAN Variants

Run:

```bash
python 03_train.py
```

The training script:

- Loads the processed data.
- Splits each dataset into 80% training data and 20% testing data.
- Trains each GAN variant for 60 epochs.
- Uses batch size 128 and latent dimension 64.
- Uses Adam optimization with learning rate 0.0002.
- Saves Generator and Discriminator checkpoints in `checkpoints/`.

The trained variants are:

```text
GAN
ACGAN
MGAN
MGAN_dis
MGAN_dis_cos
AT_MGAN
```

The AT-MGAN configuration uses:

```text
Trust score (τi): 0.8
Feature-distribution weight (α): 0.5
Cosine-loss weight (β): 0.5
```

[3][6]

## Step 5: Evaluate Models

Run:

```bash
python 04_evaluate.py
```

This script evaluates every saved discriminator checkpoint on the held-out test data and computes:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- AUC
- Matthews Correlation Coefficient
- Specificity
- False Positive Rate
- Accuracy difference relative to AT-MGAN

The results are saved as:

```text
results/ablation_unsw.csv
results/ablation_iot.csv
results/per_class_AT_MGAN_unsw.txt
results/per_class_AT_MGAN_iot.txt
```

[4]

## Step 6: Run Full Pipeline

To execute preprocessing, training, and evaluation in sequence, run:

```bash
python 05_run_all.py
```

This executes:

```text
01_preprocessing.py
03_train.py
04_evaluate.py
```

[5]

## Step 7: Run Optimized Ablation Study

For the complete optimized experiment, run:

```bash
python ablation-study2-fin.py
```

This script additionally performs:

- SMOTE oversampling for UNSW-NB15 training data.
- A training-validation split.
- Early stopping with patience of 8 epochs.
- Best-checkpoint saving based on validation accuracy.
- Generator and Discriminator loss-curve generation.
- ROC-curve generation.
- Comparative ablation tables and plots.

[7]

## Output

After execution, check:

```text
processed/     # Processed datasets
checkpoints/   # Saved GAN model weights
results/       # Ablation tables, reports, ROC curves, and loss curves
```
