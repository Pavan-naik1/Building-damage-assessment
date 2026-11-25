import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

# -------------------- CONFIGURATION --------------------
SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 100
EPOCHS = 10

DATA_DIR = Path("data/clean_dataset")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

OUT_DIR = Path("results/CNN_Clean_Training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "best_cnn_clean.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Use available CPU cores; TensorFlow will use the available device.
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)

# -------------------- VERIFY DATA --------------------
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Missing folder: {folder.resolve()}\n"
            "Expected data/clean_dataset/train, validation, and test."
        )
    for class_name in ["damage", "no_damage"]:
        if not (folder / class_name).is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {folder / class_name}"
            )

print("Dataset:", DATA_DIR.resolve())
print("Output:", OUT_DIR.resolve())

# Augmentation is applied ONLY to training images.
train_gen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 1.2],
)

eval_gen = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = train_gen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    classes=["damage", "no_damage"],
    shuffle=True,
    seed=SEED,
)

val_data = eval_gen.flow_from_directory(
    str(VAL_DIR),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    classes=["damage", "no_damage"],
    shuffle=False,
)

test_data = eval_gen.flow_from_directory(
    str(TEST_DIR),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    classes=["damage", "no_damage"],
    shuffle=False,
)

print("Class mapping:", train_data.class_indices)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)
print("Test images:", test_data.samples)

# -------------------- ORIGINAL CNN ARCHITECTURE --------------------
model = Sequential(
    [
        Input(shape=(128, 128, 3)),

        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(1024, activation="relu"),
        Dense(1024, activation="relu"),
        Dense(1, activation="sigmoid"),
    ],
    name="Base_Model_Clean",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -------------------- TRAIN --------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    CSVLogger(str(OUT_DIR / "training_log.csv")),
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

with open(OUT_DIR / "training_history.json", "w") as f:
    json.dump(
        {k: [float(x) for x in v] for k, v in history.history.items()},
        f,
        indent=2,
    )

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("CNN Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("CNN Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Binary cross-entropy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=200)
plt.close()

# -------------------- EVALUATE BEST MODEL --------------------
best_model = tf.keras.models.load_model(str(MODEL_PATH))

test_data.reset()
probabilities = best_model.predict(test_data, verbose=1).ravel()
y_true = test_data.classes
threshold = 0.50
y_pred = (probabilities >= threshold).astype(int)

metrics = {
    "model": "CNN",
    "dataset": str(DATA_DIR),
    "test_images": int(len(y_true)),
    "threshold": threshold,
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_precision": float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "macro_recall": float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "macro_f1": float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_true, probabilities)),
}

with open(OUT_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["damage", "no_damage"],
    output_dict=True,
    zero_division=0,
)

with open(OUT_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
np.savetxt(
    OUT_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["damage", "no_damage"],
)
disp.plot(cmap="Blues", values_format="d")
plt.title("CNN — Clean Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

RocCurveDisplay.from_predictions(y_true, probabilities)
plt.title("CNN — Clean Test ROC Curve")
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve.png", dpi=200)
plt.close()

# Wrongly predicted test image list
wrong_indices = np.flatnonzero(y_true != y_pred)
filenames = np.asarray(test_data.filepaths)

import csv
with open(
    OUT_DIR / "wrongly_predicted_test_images.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_path", "true_label", "predicted_label",
        "damage_probability", "threshold",
    ])
    for i in wrong_indices:
        writer.writerow([
            str(filenames[i]),
            "damage" if y_true[i] == 0 else "no_damage",
            "damage" if y_pred[i] == 0 else "no_damage",
            float(probabilities[i]),
            threshold,
        ])

print("\n========== CNN CLEAN TEST RESULTS ==========")
for key, value in metrics.items():
    print(f"{key}: {value}")
print("\nClassification report:")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["damage", "no_damage"],
    zero_division=0,
))
print("Misclassified images:", len(wrong_indices))
print("Saved results to:", OUT_DIR.resolve())
print("============================================")