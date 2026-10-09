# VERIFAI: adaptive deepfake detection prototype

This project is a research prototype for image/video deepfake classification and adversarial robustness evaluation. It uses trained PyTorch ResNet-18 binary classifiers; it does not ship pretrained deepfake weights or a dataset. A score is not proof of authenticity.

## Install

From the project directory in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a GPU-enabled PyTorch installation, use the install command recommended by [PyTorch](https://pytorch.org/get-started/locally/) for the local CUDA version before installing the remaining requirements.

## Prepare labeled data and train

Use separate training and held-out validation folders. Each must contain `fake` and `real` subfolders of labeled images:

```text
data/
  train/
    fake/
    real/
  val/
    fake/
    real/
```

Train a balanced model; optional ImageNet initialization downloads weights on first use:

```powershell
python train.py --train-dir data/train --val-dir data/val --output models/balanced.pt --profile balanced --epochs 5 --pretrained
```

Train optional low-quality and high-quality specialists as separate checkpoints. The low-quality profile uses blur and posterization augmentation. Place multiple checkpoints in the same `models` folder for the app's quality-based selector:

```powershell
python train.py --train-dir data/train --val-dir data/val --output models/low_quality.pt --profile low_quality --epochs 5 --pretrained
python train.py --train-dir data/train --val-dir data/val --output models/high_quality.pt --profile high_quality --epochs 5 --pretrained
```

Profiles describe the data augmentation used for training. Use a separate held-out evaluation set to decide whether the specialist is actually useful; profile selection alone does not establish model quality.

## Run the app

```powershell
python -m streamlit run app.py
```

The app samples up to eight evenly spaced video frames, measures image quality, selects a configured profile, runs available detectors, and withholds a verdict when models disagree, the score is ambiguous, or sensitivity to benign perturbations is high. Reported confidence is an uncalibrated score-distance estimate, not a statistical guarantee. The sensitivity probe is a warning signal, not a reliable detector of whether a particular upload was attacked.

Set `VERIFAI_MODELS_DIR` to use another checkpoint directory:

```powershell
$env:VERIFAI_MODELS_DIR = "C:\path\to\models"
python -m streamlit run app.py
```

## Evaluate FGSM and PGD robustness

Run attacks only against a labeled held-out test set. Epsilon is measured in pixel units on images scaled to `[0, 1]`:

```powershell
python evaluate.py --data-dir data/test --checkpoint models/balanced.pt --epsilons 0.005 0.01 0.03 --pgd-steps 10
```

The JSON output reports clean and attacked accuracy, precision, recall, F1, false-positive rate, ROC-AUC, and accuracy drop from clean performance. Do not tune on the held-out test set. These attacks are white-box evaluations against the selected checkpoint; they do not establish robustness to all attacks or unseen manipulations.

## Run tests

```powershell
python -m unittest discover -s tests
```
