import tensorflow as tf

m = tf.keras.models.load_model(
    "results/CNN_Clean_Training/best_cnn_clean.keras",
    compile=False
)

x = tf.random.uniform((1, 128, 128, 3))
conv_output = None

with tf.GradientTape() as tape:
    z = x

    for layer in m.layers:
        z = layer(z, training=False)

        if layer.name == "conv2d_5":
            conv_output = z

    prediction = z
    score = tf.reduce_sum(prediction)

gradients = tape.gradient(score, conv_output)

print("Prediction:", prediction.numpy())
print("Convolution output:", conv_output.shape)
print("Gradient available:", gradients is not None)

if gradients is not None:
    print("Gradient mean:", float(tf.reduce_mean(tf.abs(gradients)).numpy()))
