import tensorflow as tf

m = tf.keras.models.load_model(
    "results/CNN_Clean_Training/best_cnn_clean.keras",
    compile=False
)

last_conv = m.get_layer("conv2d_5")
grad_model = tf.keras.Model(
    inputs=m.inputs,
    outputs=[last_conv.output, m.outputs[0]]
)

x = tf.random.uniform((1, 128, 128, 3), minval=0, maxval=1)

with tf.GradientTape() as tape:
    tape.watch(x)
    conv_output, prediction = grad_model(x, training=False)
    score = tf.reduce_sum(prediction)

gradients = tape.gradient(score, conv_output)

print("Feature map:", conv_output.shape)
print("Prediction:", prediction.numpy())
print("Gradient available:", gradients is not None)

if gradients is not None:
    print("Gradient mean:", float(tf.reduce_mean(tf.abs(gradients)).numpy()))
