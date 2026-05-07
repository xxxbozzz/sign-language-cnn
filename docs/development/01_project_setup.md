# 01. Project setup

I started by keeping the project small and reproducible. The task is ASL static letter classification, so the code only needs to handle single grayscale images rather than webcam video or full sign-language translation.

The first version of the project needed four basic parts:

- a dataset folder for Kaggle CSV files
- a training script for the full experiment
- a notebook for interactive work
- output folders for figures, metrics, and models

I also fixed a random seed in the code. This matters because the project compares several CNN models. If every model used a different validation split or different random setup, the comparison would be less fair.

The main dependencies are NumPy, pandas, TensorFlow, scikit-learn, matplotlib, seaborn, and JupyterLab. I kept them in `requirements.txt` so the environment can be rebuilt.
