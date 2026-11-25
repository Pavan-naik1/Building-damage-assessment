import tensorflow as tf

model_path = r"results\EfficientNetB0_Clean_Training\best_efficientnetb0_clean.keras"

print("\nLoading saved EfficientNetB0 model...")

model = tf.keras.models.load_model(
    model_path,
    compile=False
)

print("\n========== MODEL PARAMETER COUNT ==========")

print("Model name:", model.name)
print("Total parameters:", model.count_params())
print("Trainable parameters:",
      sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
print("Non-trainable parameters:",
      sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))

print("\n===========================================")
