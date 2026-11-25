
import os
import tensorflow as tf

# Paths to the two saved CNN models in your project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "best_cnn_clean.keras"),
    os.path.join(BASE_DIR, "models", "best_cnn_optimized.keras"),
    os.path.join(
        BASE_DIR,
        "results",
        "CNN_Clean_Training",
        "best_cnn_clean.keras",
    ),
]

for path in MODEL_PATHS:
    print("\n" + "=" * 65)
    print("MODEL:", path)
    print("=" * 65)

    if not os.path.exists(path):
        print("File not found. Skipping.")
        continue

    try:
        model = tf.keras.models.load_model(path, compile=False)

        print("\nInput shape:", model.input_shape)
        print("Output shape:", model.output_shape)

        print("\nLayer-by-layer summary:")
        model.summary()

        total_params = model.count_params()
        trainable_params = sum(
            int(tf.size(variable).numpy())
            for variable in model.trainable_variables
        )

        print("\nPARAMETER COUNT")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable:        {total_params - trainable_params:,}")
        print(f"Total in millions:    {total_params / 1_000_000:.3f} M")

    except Exception as error:
        print("Could not load this model:")
        print(type(error).__name__, error)

print("\nFinished checking available CNN models.")
