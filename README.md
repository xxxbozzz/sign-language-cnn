# Sign Language CNN Classifier

English | [中文](#中文说明)

This repository contains the code, saved models, evaluation artifacts, and development notes for a CNN-based American Sign Language letter classifier using the Sign Language MNIST dataset.

The project compares three models: a simple baseline CNN, a tuned baseline CNN, and an enhanced VGG-style CNN. The enhanced model achieved 99.39% test accuracy on the official Kaggle test set.

## Technology Stack

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook
- Git / GitHub

## Repository Contents

- `train_and_generate_report.py`: full training and evaluation pipeline.
- `sign_language_cnn_assignment.ipynb`: notebook version of the workflow.
- `docs/development/`: step-by-step notes explaining how each part of the code was built.
- `data/README.md`: dataset download instructions.
- `figures/`: generated plots for class distribution, training curves, confusion matrix, and sample predictions.
- `outputs/`: comparison tables, classification report, confusion matrix CSV, and trained `.h5` models.

Final coursework report files are not included in this public repository. This repository focuses on the code process and experiment artifacts.

## Dataset

The raw Kaggle CSV files are not committed. Download the Sign Language MNIST dataset from Kaggle:

https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Then place these two files in `data/`:

```text
data/sign_mnist_train.csv
data/sign_mnist_test.csv
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is usually easier to install with Python 3.10, 3.11, or 3.12.

## Run the Experiment

```bash
python train_and_generate_report.py
```

The script loads the dataset, preprocesses the images, trains the three CNN experiments, saves the trained models, and writes evaluation outputs into `outputs/` and `figures/`.

## Best Result

The enhanced VGG-style CNN was the best model in this experiment:

- Validation accuracy: 100.00%
- Test accuracy: 99.39%
- Macro F1-score: 0.9939
- Weighted F1-score: 0.9939

The full comparison is saved in `outputs/model_comparison.csv`.

## Development Notes

The coding process is documented in separate files instead of one long document:

- `docs/development/01_project_setup.md`
- `docs/development/02_data_pipeline.md`
- `docs/development/03_baseline_cnn.md`
- `docs/development/04_enhanced_vgg_model.md`
- `docs/development/05_training_experiments.md`
- `docs/development/06_evaluation_outputs.md`
- `docs/development/07_saved_models.md`

---

# 中文说明

本仓库包含一个基于 CNN 的美国手语字母识别项目代码、已训练模型、评估结果和代码制作流程说明。数据集使用 Kaggle 的 Sign Language MNIST。

项目对比了三个模型：基础 CNN、调参后的基础 CNN、增强版 VGG-style CNN。最终增强模型在官方 Kaggle test set 上达到 99.39% test accuracy。

## 技术栈

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook
- Git / GitHub

## 仓库内容

- `train_and_generate_report.py`：完整训练和评估流程。
- `sign_language_cnn_assignment.ipynb`：Notebook 版本的实验流程。
- `docs/development/`：分阶段说明代码是怎么一步步写出来的。
- `data/README.md`：数据集下载说明。
- `figures/`：类别分布、训练曲线、混淆矩阵和预测样例图。
- `outputs/`：模型对比表、classification report、confusion matrix CSV 和训练好的 `.h5` 模型。

本公开仓库不包含最终 coursework report 文件。这里主要展示代码流程、实验过程和可复现产物。

## 数据集

原始 Kaggle CSV 文件没有提交到仓库。请从 Kaggle 下载 Sign Language MNIST：

https://www.kaggle.com/datasets/datamunge/sign-language-mnist

下载后，把下面两个文件放到 `data/`：

```text
data/sign_mnist_train.csv
data/sign_mnist_test.csv
```

## 环境配置

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow 通常在 Python 3.10、3.11 或 3.12 上更容易安装。

## 运行实验

```bash
python train_and_generate_report.py
```

脚本会读取数据、完成预处理、训练三个 CNN 实验、保存模型，并把评估结果写入 `outputs/` 和 `figures/`。

## 最佳结果

本项目中表现最好的模型是增强版 VGG-style CNN：

- Validation accuracy: 100.00%
- Test accuracy: 99.39%
- Macro F1-score: 0.9939
- Weighted F1-score: 0.9939

完整对比结果保存在 `outputs/model_comparison.csv`。

## 代码制作流程

代码制作流程没有写成一个总文档，而是拆成多个阶段文件：

- `docs/development/01_project_setup.md`
- `docs/development/02_data_pipeline.md`
- `docs/development/03_baseline_cnn.md`
- `docs/development/04_enhanced_vgg_model.md`
- `docs/development/05_training_experiments.md`
- `docs/development/06_evaluation_outputs.md`
- `docs/development/07_saved_models.md`
