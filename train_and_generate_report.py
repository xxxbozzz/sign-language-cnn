import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
OUT_DIR = PROJECT_DIR / "outputs"
for directory in (FIG_DIR, OUT_DIR):
    directory.mkdir(exist_ok=True)

TRAIN_CSV = DATA_DIR / "sign_mnist_train.csv"
TEST_CSV = DATA_DIR / "sign_mnist_test.csv"

ORIGINAL_LABELS = [i for i in range(25) if i != 9]
CLASS_NAMES = [chr(ord("A") + i) for i in ORIGINAL_LABELS]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(ORIGINAL_LABELS)}


def load_sign_mnist_csv(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")
    df = pd.read_csv(csv_path)
    y_original = df["label"].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    x = pixels.reshape((-1, 28, 28, 1)) / 255.0
    y = np.array([LABEL_TO_INDEX[int(label)] for label in y_original], dtype=np.int64)
    return x, y, y_original


def make_optimizer(name="adam", learning_rate=1e-3):
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def compile_model(model, optimizer_name="adam", learning_rate=1e-3):
    model.compile(
        optimizer=make_optimizer(optimizer_name, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline_cnn(learning_rate=1e-3, optimizer_name="adam", dropout_rate=0.30):
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="baseline_cnn")
    return compile_model(model, optimizer_name, learning_rate)


def conv_bn_relu(x, filters, l2_strength):
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_strength),
    )(x)
    return layers.BatchNormalization()(x)


def build_vgg_style_cnn(
    learning_rate=1e-3,
    optimizer_name="adam",
    dropout_rate=0.40,
    l2_strength=1e-4,
):
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs
    for filters in (32, 64, 128):
        x = conv_bn_relu(x, filters, l2_strength)
        x = conv_bn_relu(x, filters, l2_strength)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate / 2 if filters < 128 else dropout_rate)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_strength),
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="vgg_style_cnn")
    return compile_model(model, optimizer_name, learning_rate)


def save_class_distribution(y_train_full):
    counts = pd.Series(y_train_full).value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in counts.index]
    plt.figure(figsize=(11, 4))
    sns.barplot(x=labels, y=counts.values, color="#4c78a8")
    plt.title("Training Class Distribution")
    plt.xlabel("ASL letter")
    plt.ylabel("Number of images")
    plt.tight_layout()
    path = FIG_DIR / "class_distribution.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def save_sample_grid(X_train_full, y_train_full):
    selected = []
    for class_idx in range(NUM_CLASSES):
        matches = np.where(y_train_full == class_idx)[0]
        if len(matches) > 0:
            selected.append(matches[0])
    fig, axes = plt.subplots(4, 6, figsize=(9, 6))
    for ax, image_idx in zip(axes.flat, selected):
        ax.imshow(X_train_full[image_idx].squeeze(), cmap="gray")
        ax.set_title(CLASS_NAMES[y_train_full[image_idx]])
        ax.axis("off")
    plt.suptitle("Example Image from Each Class", y=1.02)
    plt.tight_layout()
    path = FIG_DIR / "sample_images.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title(f"{model_name}: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Training loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title(f"{model_name}: Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    plt.tight_layout()
    path = FIG_DIR / f"training_curves_{model_name}.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def train_one_experiment(config, X_train, y_train, X_val, y_val, X_test, y_test):
    tf.keras.backend.clear_session()
    kwargs = {
        "learning_rate": config["learning_rate"],
        "optimizer_name": config["optimizer_name"],
        "dropout_rate": config["dropout_rate"],
    }
    if "l2_strength" in config:
        kwargs["l2_strength"] = config["l2_strength"]
    model = config["builder"](**kwargs)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )
    training_time = time.time() - start_time
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    curve_path = plot_training_curves(history, config["name"])
    model_path = OUT_DIR / f"{config['name']}.h5"
    model.save(model_path)
    row = {
        "model": config["name"],
        "architecture": config["architecture"],
        "optimizer": config["optimizer_name"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "dropout_rate": config["dropout_rate"],
        "l2_strength": config.get("l2_strength", 0.0),
        "epochs_run": len(history.history["loss"]),
        "parameters": model.count_params(),
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "curve_path": str(curve_path),
        "model_path": str(model_path),
    }
    return model, history, row


def save_best_evaluation(best_model, best_name, X_test, y_test):
    y_prob = best_model.predict(X_test, batch_size=128, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = OUT_DIR / "classification_report_best.csv"
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix: {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    cm_path = FIG_DIR / "confusion_matrix_best.png"
    plt.savefig(cm_path, dpi=180, bbox_inches="tight")
    plt.close()

    rng = np.random.default_rng(SEED)
    sample_indices = rng.choice(len(X_test), size=12, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax, idx in zip(axes.flat, sample_indices):
        true_label = CLASS_NAMES[y_test[idx]]
        predicted_label = CLASS_NAMES[y_pred[idx]]
        color = "green" if true_label == predicted_label else "red"
        ax.imshow(X_test[idx].squeeze(), cmap="gray")
        ax.set_title(f"True: {true_label} | Pred: {predicted_label}", color=color, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"Sample Predictions: {best_name}", y=1.02)
    plt.tight_layout()
    pred_path = FIG_DIR / "sample_predictions_best.png"
    plt.savefig(pred_path, dpi=180, bbox_inches="tight")
    plt.close()

    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path_csv = OUT_DIR / "confusion_matrix_best.csv"
    cm_df.to_csv(cm_path_csv)

    class_report_text_path = OUT_DIR / "classification_report_best.txt"
    class_report_text_path.write_text(
        classification_report(y_test, y_pred, target_names=CLASS_NAMES),
        encoding="utf-8",
    )

    return {
        "classification_report": str(report_path),
        "classification_report_text": str(class_report_text_path),
        "confusion_matrix": str(cm_path),
        "confusion_matrix_csv": str(cm_path_csv),
        "sample_predictions": str(pred_path),
        "macro_f1": float(report_df.loc["macro avg", "f1-score"]),
        "weighted_f1": float(report_df.loc["weighted avg", "f1-score"]),
    }


def main():
    print("TensorFlow:", tf.__version__)
    X_train_full, y_train_full, _ = load_sign_mnist_csv(TRAIN_CSV)
    X_test, y_test, _ = load_sign_mnist_csv(TEST_CSV)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=SEED,
        stratify=y_train_full,
    )
    print("Training split:", X_train.shape, y_train.shape)
    print("Validation split:", X_val.shape, y_val.shape)
    print("Test set:", X_test.shape, y_test.shape)
    class_distribution_path = save_class_distribution(y_train_full)
    sample_grid_path = save_sample_grid(X_train_full, y_train_full)

    experiments = [
        {
            "name": "baseline_adam_lr1e-3_bs128",
            "architecture": "Baseline CNN",
            "builder": build_baseline_cnn,
            "optimizer_name": "adam",
            "learning_rate": 1e-3,
            "batch_size": 128,
            "dropout_rate": 0.30,
            "epochs": 8,
        },
        {
            "name": "baseline_adam_lr5e-4_bs256",
            "architecture": "Baseline CNN tuned LR/batch",
            "builder": build_baseline_cnn,
            "optimizer_name": "adam",
            "learning_rate": 5e-4,
            "batch_size": 256,
            "dropout_rate": 0.30,
            "epochs": 8,
        },
        {
            "name": "vgg_style_adam_lr1e-3_bs128",
            "architecture": "Enhanced VGG-style CNN",
            "builder": build_vgg_style_cnn,
            "optimizer_name": "adam",
            "learning_rate": 1e-3,
            "batch_size": 128,
            "dropout_rate": 0.40,
            "l2_strength": 1e-4,
            "epochs": 10,
        },
    ]

    models = {}
    results = []
    for config in experiments:
        print("\n" + "=" * 80)
        print("Training:", config["name"])
        model, history, row = train_one_experiment(config, X_train, y_train, X_val, y_val, X_test, y_test)
        models[config["name"]] = model
        results.append(row)

    comparison_df = pd.DataFrame(results).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    comparison_path = OUT_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(comparison_df)

    best_name = comparison_df.loc[0, "model"]
    eval_artifacts = save_best_evaluation(models[best_name], best_name, X_test, y_test)
    summary = {
        "best_model": best_name,
        "comparison_table": str(comparison_path),
        "class_distribution": str(class_distribution_path),
        "sample_images": str(sample_grid_path),
        **eval_artifacts,
    }
    summary_path = OUT_DIR / "report_artifacts.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
