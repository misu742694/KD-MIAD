# Federated Learning Membership Inference Defense

This project implements a federated learning defense combining entropy-regularized teacher training, CVAE-based knowledge distillation, and contribution-aware aggregation.

## Quick Start

Run the following commands **in order** from the project root:

```bash
# 1. Train the CVAE and generate synthetic data
python train-vae.py

# 2. Run federated learning with distillation
python fl_basic.py
```

To regenerate synthetic data using an existing CVAE checkpoint:

```bash
python train-vae.py --mode generate --samples-per-class 100 --num-batches 1
```

Before running, configure the dataset and its path in `constants.py` and ensure the required directories exist.

## Files

| File | Description |
| --- | --- |
| `train-vae.py` | Trains a conditional VAE and generates labeled synthetic data for student distillation. |
| `fl_basic.py` | Main entry point for federated training with distillation. |
| `organizer.py` | Coordinates clients, training rounds, aggregation, evaluation, and model saving. |
| `models.py` | Defines dataset-specific models, teacher/student training, distillation, and client behavior. |
| `aggregator.py` | Collects client updates and implements aggregation strategies, including contribution-aware aggregation. |
| `data_reader.py` | Loads datasets into client training and test sets. |
| `constants.py` | Stores training hyperparameters, aggregation settings, and other configuration. |


