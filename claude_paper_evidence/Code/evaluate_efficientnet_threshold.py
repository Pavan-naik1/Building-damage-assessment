import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "clean_dataset"

VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

MODEL_PATH = BASE_DIR / "models" / "best_efficientnetb0.keras"
RESULTS_DIR = BASE_DIR / "results" / "EfficientNetB0_Clean"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
CLASS_NAMES = ["damage", "no_damage"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("EFFICIENTNETB0 CLEAN-DATASET EVALUATION")
print("=" * 68)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VAL_DIR.exists() or not TEST_DIR.exists():
    raise FileNotFoundError("Clean validation/test folders were not found.")

# ============================================================
# LOAD MODEL; REMOVE NULL QUANTIZATION CONFIG FOR COMPATIBILITY
# The original model file is not modified.
# ============================================================

print("\nLoading EfficientNetB0 model...")

with tempfile.TemporaryDirectory() as temp_dir:
    compatible_model_path = Path(temp_dir) / "efficientnet_compatible.keras"

    with zipfile.ZipFile(MODEL_PATH, "r") as source_zip:
        with zipfile.ZipFile(
            compatible_model_path, "w", zipfile.ZIP_DEFLATED
        ) as target_zip:
            for item in source_zip.infolist():
                content = source_zip.read(item.filename)

                if item.filename == "config.json":
                    config = json.loads(content.decode("utf-8"))

                    def remove_null_quantization_config(obj):
                        if isinstance(obj, dict):
                            if obj.get("quantization_config", "missing") is None:
                                obj.pop("quantization_config")

                            for value in list(obj.values()):
                                remove_null_quantization_config(value)

                        elif isinstance(obj, list):
                            for value in obj:
                                remove_null_quantization_config(value)

                    remove_null_quantization_config(config)
                    content = json.dumps(config).encode("utf-8")

                target_zip.writestr(item, content)

    model = tf.keras.models.load_model(
        compatible_model_path,
        compile=False,
    )

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# DATASET LOADING
# Preserve class names separately before prefetching.
# EfficientNet preprocessing assumed to be inside the model:
# images are fed in the 0-255 range.
# ============================================================

def load_dataset(directory):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    detected_classes = list(dataset.class_names)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, detected_classes


def get_predictions(dataset):
    true_values = []
    probabilities = []

    for images, labels in dataset:
        batch_probabilities = model.predict(images, verbose=0).reshape(-1)

        true_values.extend(labels.numpy().reshape(-1).astype(int))
        probabilities.extend(batch_probabilities.astype(float))

    y_true = np.asarray(true_values, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)

    if len(y_true) == 0:
        raise ValueError("No images were loaded.")

    if not np.isfinite(y_prob).all():
        raise ValueError("Model predictions contain NaN or infinity.")

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            "Model output is outside [0, 1]. Check the output activation."
        )

    return y_true, y_prob


print("\nLoading clean validation dataset...")
val_dataset, val_classes = load_dataset(VAL_DIR)
print("Validation classes:", val_classes)

print("\nLoading clean test dataset...")
test_dataset, test_classes = load_dataset(TEST_DIR)
print("Test classes:", test_classes)

# ============================================================
# SELECT THRESHOLD USING VALIDATION MACRO-F1
# Do not use test labels to select the threshold.
# ============================================================

print("\nPredicting validation dataset...")
y_val, p_val = get_predictions(val_dataset)

if len(np.unique(y_val)) != 2:
    raise ValueError("Validation dataset must contain both classes.")

threshold_rows = []

for threshold in np.arange(0.10, 0.901, 0.01):
    val_pred = (p_val >= threshold).astype(int)

    threshold_rows.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_val, val_pred),
        "macro_precision": precision_score(
            y_val, val_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_val, val_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, val_pred, average="macro", zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    RESULTS_DIR / "validation_threshold_search.csv",
    index=False,
)

best_f1 = threshold_df["macro_f1"].max()
tied = threshold_df[
    np.isclose(threshold_df["macro_f1"], best_f1)
].copy()
tied["distance_from_050"] = (tied["threshold"] - 0.50).abs()

best_threshold = float(
    tied.sort_values("distance_from_050").iloc[0]["threshold"]
)

with open(RESULTS_DIR / "selected_threshold.json", "w", encoding="utf-8") as f:
    json.dump({
        "selection_dataset": "clean_dataset/validation",
        "selection_metric": "macro_f1",
        "selected_threshold": best_threshold,
        "validation_macro_f1": float(best_f1),
    }, f, indent=4)

print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Validation macro-F1: {best_f1:.4f}")

# ============================================================
# FINAL CLEAN TEST EVALUATION
# ============================================================

print("\nPredicting clean test dataset...")
y_true, y_prob = get_predictions(test_dataset)
y_pred = (y_prob >= best_threshold).astype(int)

accuracy = accuracy_score(y_true, y_pred)
macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_true, y_pred, average="weighted", zero_division=0
)

roc_auc = (
    roc_auc_score(y_true, y_prob)
    if len(np.unique(y_true)) == 2 else None
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

metrics = {
    "model": "EfficientNetB0",
    "dataset": "clean_dataset/test",
    "test_samples": int(len(y_true)),
    "class_order": CLASS_NAMES,
    "threshold": best_threshold,
    "threshold_selection_dataset": "clean_dataset/validation",
    "threshold_selection_metric": "macro_f1",
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "roc_auc": float(roc_auc) if roc_auc is not None else None,
    "damage_precision": float(report["damage"]["precision"]),
    "damage_recall": float(report["damage"]["recall"]),
    "damage_f1": float(report["damage"]["f1-score"]),
    "no_damage_precision": float(report["no_damage"]["precision"]),
    "no_damage_recall": float(report["no_damage"]["recall"]),
    "no_damage_f1": float(report["no_damage"]["f1-score"]),
    "confusion_matrix_labels": CLASS_NAMES,
    "confusion_matrix": cm.tolist(),
}

with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

with open(
    RESULTS_DIR / "classification_report.json", "w", encoding="utf-8"
) as f:
    json.dump(report, f, indent=4)

pd.DataFrame(
    cm,
    index=[f"Actual_{name}" for name in CLASS_NAMES],
    columns=[f"Predicted_{name}" for name in CLASS_NAMES],
).to_csv(RESULTS_DIR / "confusion_matrix.csv")

# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest")
fig.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(2),
    yticks=np.arange(2),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted label",
    ylabel="Actual label",
    title=f"EfficientNetB0 Confusion Matrix (threshold={best_threshold:.2f})",
)

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            fontsize=12,
        )

fig.tight_layout()
fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# ============================================================
# ROC CURVE
# ============================================================

if roc_auc is not None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"EfficientNetB0 (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="EfficientNetB0 ROC Curve — Clean Test Set",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

# ============================================================
# MISCLASSIFIED IMAGE LIST
# ============================================================

image_paths = []

for class_name in CLASS_NAMES:
    class_dir = TEST_DIR / class_name

    paths = sorted(
        p for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    )
    image_paths.extend(paths)

if len(image_paths) != len(y_true):
    raise RuntimeError(
        f"Image/prediction mismatch: {len(image_paths)} paths, "
        f"{len(y_true)} predictions."
    )

wrong_rows = []

for idx in np.flatnonzero(y_true != y_pred):
    wrong_rows.append({
        "image_path": str(image_paths[idx].resolve()),
        "actual_class": CLASS_NAMES[int(y_true[idx])],
        "predicted_class": CLASS_NAMES[int(y_pred[idx])],
        "predicted_no_damage_probability": float(y_prob[idx]),
        "threshold": best_threshold,
    })

pd.DataFrame(
    wrong_rows,
    columns=[
        "image_path",
        "actual_class",
        "predicted_class",
        "predicted_no_damage_probability",
        "threshold",
    ],
).to_csv(
    RESULTS_DIR / "wrongly_predicted_test_images.csv",
    index=False,
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 68)
print("EFFICIENTNETB0 CLEAN TEST RESULTS")
print("=" * 68)

print(f"Test samples       : {len(y_true)}")
print(f"Selected threshold : {best_threshold:.2f}")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro precision    : {macro_precision:.4f}")
print(f"Macro recall       : {macro_recall:.4f}")
print(f"Macro F1-score     : {macro_f1:.4f}")
print(f"Weighted F1-score  : {weighted_f1:.4f}")
print(
    f"ROC-AUC            : {roc_auc:.4f}"
    if roc_auc is not None else "ROC-AUC            : N/A"
)

print("\nPer-class report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
))

print("\nConfusion matrix (rows=actual, columns=predicted):")
print(cm)
print(f"\nMisclassified images: {len(wrong_rows)}")
print("Results saved to:", RESULTS_DIR)
print("\nEvaluation completed successfully.")