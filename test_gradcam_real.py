import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_path = r"data/clean_dataset/test/damage/000325ff968032d6a4056f49ee35ebb3f2b9782d75c996c134f9803215884cd8.jpeg"

m = tf.keras.models.load_model(
    "results/CNN_Clean_Training/best_cnn_clean.keras",
    compile=False
)

image = Image.open(image_path).convert("RGB")
resized = image.resize((128, 128), resample=Image.Resampling.NEAREST)
x = np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, axis=0)
x = tf.convert_to_tensor(x)

with tf.GradientTape() as tape:
    z = x
    conv_output = None

    for layer in m.layers:
        z = layer(z, training=False)
        if layer.name == "conv2d_5":
            conv_output = z

    prediction = z
    p = prediction[0, 0]
    score = tf.where(p >= 0.5, p, 1.0 - p)

gradients = tape.gradient(score, conv_output)

print("Actual class: damage")
print("Predicted no-damage probability:", float(p.numpy()))
print("Gradient available:", gradients is not None)

if gradients is not None:
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(conv_output[0] * weights[0], axis=-1)
    cam = tf.nn.relu(cam).numpy()

    print("Gradient mean:", float(tf.reduce_mean(tf.abs(gradients)).numpy()))
    print("Heatmap minimum:", float(cam.min()))
    print("Heatmap maximum:", float(cam.max()))
    print("Heatmap standard deviation:", float(cam.std()))

    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = tf.image.resize(cam[None, ..., None], (128, 128)).numpy()[0, ..., 0]

        plt.figure(figsize=(6, 6))
        plt.imshow(resized)
        plt.imshow(cam, cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.savefig("results/gradcam_real_test_diagnostic.png",
                    dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close()
        print("Heatmap saved: results/gradcam_real_test_diagnostic.png")
    else:
        print("Heatmap is constant; no useful localization map was produced.")
