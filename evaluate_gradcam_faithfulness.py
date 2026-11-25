
from pathlib import Path
import json
import csv

import numpy as np
import tensorflow as tf
from PIL import Image


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_PATH = Path(
    "results/CNN_Clean_Training/best_cnn_clean.keras"
)

TEST_DIR = Path(
    "data/clean_dataset/test"
)

OUTPUT_DIR = Path(
    "results/GradCAM_Faithfulness"
)

IMAGE_SIZE = (128, 128)

SAMPLES_PER_CLASS = 25

STEPS = 10

BASELINE_VALUE = 0.5

SEED = 42

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

np.random.seed(SEED)


# ==================================================
# 2. LOAD CNN MODEL
# ==================================================

print("\nLoading CNN model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")

print("Input shape:", model.inputs[0].shape)

print("Output shape:", model.outputs[0].shape)


# ==================================================
# 3. REBUILD GRAPH FOR GRAD-CAM
# ==================================================

print("\nBuilding Grad-CAM model...")

conv_layer = model.get_layer("conv2d_5")

# Start with the original model input.
x = model.inputs[0]

feature_maps = None

# Reapply each layer in the original model.
# This explicitly connects the final convolution
# output to the final sigmoid prediction.

for layer in model.layers:

    x = layer(x)

    if layer.name == "conv2d_5":
        feature_maps = x


if feature_maps is None:
    raise RuntimeError(
        "Could not locate conv2d_5 output."
    )


grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        feature_maps,
        x
    ]
)

print("Grad-CAM graph created.")

print("Target convolution layer:", conv_layer.name)


# ==================================================
# 4. LOAD AND PREPROCESS IMAGE
# ==================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    return image_array


# ==================================================
# 5. GENERATE GENUINE GRAD-CAM
# ==================================================

def generate_gradcam(image):

    input_tensor = tf.convert_to_tensor(
        image[np.newaxis, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        feature_maps, predictions = grad_model(
            input_tensor,
            training=False
        )

        probability = predictions[0, 0]

        # Class encoding:
        # 0 = damage
        # 1 = no_damage

        predicted_class = tf.cast(
            probability >= 0.5,
            tf.int32
        )

        # Explain the predicted class.
        class_score = tf.where(
            predicted_class == 1,
            probability,
            1.0 - probability
        )

    gradients = tape.gradient(
        class_score,
        feature_maps
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients unavailable. "
            "The convolution output is not connected "
            "to the prediction."
        )

    # Average gradients over spatial dimensions.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted sum of convolution feature maps.
    heatmap = tf.reduce_sum(
        feature_maps *
        weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    # Apply ReLU.
    heatmap = tf.nn.relu(
        heatmap[0]
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum.numpy()) <= 1e-10:

        raise RuntimeError(
            "Generated Grad-CAM heatmap is zero."
        )

    heatmap = heatmap / maximum

    # Resize heatmap to input image dimensions.
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE
    )

    heatmap = heatmap[..., 0]

    return (
        heatmap.numpy(),
        int(predicted_class.numpy()),
        float(class_score.numpy())
    )


# ==================================================
# 6. GET CLASS CONFIDENCE
# ==================================================

def get_class_confidence(
    image,
    class_index
):

    prediction = model(
        image[np.newaxis, ...],
        training=False
    )[0, 0]

    probability = float(
        prediction.numpy()
    )

    # Model output is probability of no_damage.
    if class_index == 1:

        return probability

    return 1.0 - probability


# ==================================================
# 7. DELETION AND INSERTION EVALUATION
# ==================================================

def faithfulness_test(
    image,
    heatmap,
    class_index
):

    flat_heatmap = heatmap.reshape(-1)

    # Rank pixels from highest to lowest saliency.
    ranking = np.argsort(
        flat_heatmap
    )[::-1]

    total_pixels = len(ranking)

    deletion_scores = []

    insertion_scores = []

    baseline = np.full_like(
        image,
        BASELINE_VALUE
    )

    for step in range(STEPS + 1):

        fraction = step / STEPS

        number = int(
            fraction * total_pixels
        )

        selected = ranking[:number]

        mask = np.zeros(
            total_pixels,
            dtype=bool
        )

        mask[selected] = True

        mask = mask.reshape(
            IMAGE_SIZE
        )

        mask = mask[..., np.newaxis]

        # Deletion:
        # Replace the most salient pixels.
        deleted_image = np.where(
            mask,
            BASELINE_VALUE,
            image
        )

        # Insertion:
        # Restore the most salient pixels.
        inserted_image = np.where(
            mask,
            image,
            baseline
        )

        deletion_confidence = get_class_confidence(
            deleted_image,
            class_index
        )

        insertion_confidence = get_class_confidence(
            inserted_image,
            class_index
        )

        deletion_scores.append(
            deletion_confidence
        )

        insertion_scores.append(
            insertion_confidence
        )

    x_values = np.linspace(
        0,
        1,
        STEPS + 1
    )

    deletion_auc = float(
        np.trapezoid(
            deletion_scores,
            x_values
        )
    )

    insertion_auc = float(
        np.trapezoid(
            insertion_scores,
            x_values
        )
    )

    return (
        deletion_auc,
        insertion_auc
    )


# ==================================================
# 8. SELECT BALANCED TEST IMAGES
# ==================================================

print("\nSelecting test images...")

class_names = [
    "damage",
    "no_damage"
]

selected_images = []

for class_index, class_name in enumerate(class_names):

    class_dir = TEST_DIR / class_name

    if not class_dir.exists():

        raise FileNotFoundError(
            f"Missing test folder: {class_dir}"
        )

    image_paths = []

    for extension in (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ):

        image_paths.extend(
            class_dir.glob(extension)
        )

    image_paths = sorted(
        image_paths
    )

    if len(image_paths) < SAMPLES_PER_CLASS:

        raise ValueError(
            f"Not enough images in {class_dir}"
        )

    rng = np.random.default_rng(
        SEED + class_index
    )

    chosen_indices = rng.choice(
        len(image_paths),
        size=SAMPLES_PER_CLASS,
        replace=False
    )

    for index in chosen_indices:

        selected_images.append(
            (
                class_name,
                image_paths[index]
            )
        )

print(
    "Selected images:",
    len(selected_images)
)


# ==================================================
# 9. CHECK GRADIENTS BEFORE FULL EVALUATION
# ==================================================

print("\nChecking Grad-CAM gradients...")

test_image = load_image(
    selected_images[0][1]
)

test_heatmap, test_class, test_confidence = generate_gradcam(
    test_image
)

print("Gradient check passed.")

print("Test predicted class:", class_names[test_class])

print("Test confidence:", round(test_confidence, 4))

print("Heatmap shape:", test_heatmap.shape)


# ==================================================
# 10. RUN FAITHFULNESS EVALUATION
# ==================================================

print("\nStarting faithfulness evaluation...")

results = []

for index, (
    true_class,
    image_path
) in enumerate(
    selected_images,
    start=1
):

    print(
        f"\n[{index}/{len(selected_images)}] "
        f"{image_path.name}"
    )

    try:

        image = load_image(
            image_path
        )

        heatmap, predicted_class, original_confidence = (
            generate_gradcam(image)
        )

        deletion_auc, insertion_auc = faithfulness_test(
            image,
            heatmap,
            predicted_class
        )

        predicted_label = class_names[
            predicted_class
        ]

        result = {

            "image": image_path.name,

            "true_class": true_class,

            "predicted_class": predicted_label,

            "original_confidence": original_confidence,

            "deletion_auc": deletion_auc,

            "insertion_auc": insertion_auc,

            "gradcam_status": "genuine_gradcam"

        }

        results.append(
            result
        )

        print(
            "True class:",
            true_class
        )

        print(
            "Predicted class:",
            predicted_label
        )

        print(
            "Original confidence:",
            round(original_confidence, 4)
        )

        print(
            "Deletion AUC:",
            round(deletion_auc, 4)
        )

        print(
            "Insertion AUC:",
            round(insertion_auc, 4)
        )

    except Exception as error:

        print(
            "ERROR:",
            str(error)
        )


# ==================================================
# 11. SAVE CSV RESULTS
# ==================================================

if not results:

    raise RuntimeError(
        "No images were evaluated successfully. "
        "Check the Grad-CAM error messages."
    )


csv_path = (
    OUTPUT_DIR /
    "gradcam_faithfulness_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ==================================================
# 12. CALCULATE SUMMARY STATISTICS
# ==================================================

deletion_values = [
    row["deletion_auc"]
    for row in results
]

insertion_values = [
    row["insertion_auc"]
    for row in results
]

summary = {

    "model": "CNN",

    "gradcam_layer": "conv2d_5",

    "evaluated_images": len(results),

    "requested_images": len(selected_images),

    "samples_per_class": SAMPLES_PER_CLASS,

    "steps": STEPS,

    "baseline_value": BASELINE_VALUE,

    "deletion_auc_mean": float(
        np.mean(deletion_values)
    ),

    "deletion_auc_std": float(
        np.std(deletion_values)
    ),

    "insertion_auc_mean": float(
        np.mean(insertion_values)
    ),

    "insertion_auc_std": float(
        np.std(insertion_values)
    ),

    "interpretation": (
        "Lower deletion AUC and higher insertion AUC "
        "are generally consistent with stronger faithfulness "
        "under this perturbation procedure. "
        "These measurements do not establish spatial localization."
    )

}


json_path = (
    OUTPUT_DIR /
    "gradcam_faithfulness_summary.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ==================================================
# 13. FINAL REPORT
# ==================================================

print("\n" + "=" * 55)

print("GRAD-CAM FAITHFULNESS EVALUATION COMPLETED")

print("=" * 55)

print(
    "Successfully evaluated:",
    len(results),
    "/",
    len(selected_images)
)

print(
    "Deletion AUC:",
    round(summary["deletion_auc_mean"], 4),
    "+/-",
    round(summary["deletion_auc_std"], 4)
)

print(
    "Insertion AUC:",
    round(summary["insertion_auc_mean"], 4),
    "+/-",
    round(summary["insertion_auc_std"], 4)
)

print("\nSaved files:")

print(csv_path)

print(json_path)

print("\nEvaluation completed successfully.")
