# Sign Language CNN Classifier

This repository contains the code and development notes for a CNN-based American Sign Language letter classifier using the Sign Language MNIST dataset.

The project compares a simple baseline CNN, a tuned baseline, and an enhanced VGG-style CNN. The final model reached 99.39% test accuracy on the official Kaggle test set. The repository focuses on the coding process, experiment pipeline, saved models, and evaluation artifacts.

## Repository contents

- `train_and_generate_report.py`: full training and evaluation pipeline.
- `sign_language_cnn_assignment.ipynb`: notebook version of the same workflow.
- `docs/development/`: step-by-step notes explaining how each part of the code was built.
- `data/README.md`: dataset download instructions.
- `figures/`: generated plots used for evaluation.
- `outputs/`: comparison tables, classification report, confusion matrix, and trained `.h5` models.

## Dataset

The raw Kaggle CSV files are not committed to this repository. Download the Sign Language MNIST dataset from Kaggle and place the two CSV files in `data/`:

```text
data/sign_mnist_train.csv
data/sign_mnist_test.csv
```

Dataset source: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is usually easier to install with Python 3.10, 3.11, or 3.12.

## Run the experiment

```bash
python train_and_generate_report.py
```

The script loads the dataset, preprocesses images, trains three CNN experiments, saves model files, and writes evaluation outputs into `outputs/` and `figures/`.

## Best result

The enhanced VGG-style CNN was the best model in this experiment:

- Validation accuracy: 100.00%
- Test accuracy: 99.39%
- Macro F1-score: 0.9939
- Weighted F1-score: 0.9939

The full comparison is saved in `outputs/model_comparison.csv`.
