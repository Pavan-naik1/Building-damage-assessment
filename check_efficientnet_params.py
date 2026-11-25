import tensorflow as tf

model_path = r"models\best_efficientnetb0.keras"

model = tf.keras.models.load_model(model_path, compile=False)

print("\n===== MODEL SUMMARY =====")
model.summary()

total = model.count_params()
trainable = sum(
    tf.keras.backend.count_params(w)
    for w in model.trainable_weights
)
non_trainable = sum(
    tf.keras.backend.count_params(w)
    for w in model.non_trainable_weights
)

print("\n===== PARAMETER COUNTS =====")
print("Model name:", model.name)
print("Total parameters:", f"{total:,}")
print("Trainable parameters:", f"{trainable:,}")
print("Non-trainable parameters:", f"{non_trainable:,}")

print("\n===== LAYER TRAINABILITY =====")
for layer in model.layers:
    print(
        layer.name,
        "| trainable:", layer.trainable,
        "| parameters:", f"{layer.count_params():,}"
    )
    if isinstance(layer, tf.keras.Model):
        print("Nested model trainable:", layer.trainable)
        print("Nested model trainable parameters:",
              sum(tf.keras.backend.count_params(w)
                  for w in layer.trainable_weights))
