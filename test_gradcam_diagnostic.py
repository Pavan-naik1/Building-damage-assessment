import tensorflow as tf

m = tf.keras.models.load_model(
    "results/CNN_Clean_Training/best_cnn_clean.keras",
    compile=False
)

layer = m.get_layer("conv2d_5")
gm = tf.keras.Model(m.inputs, [layer.output, m.outputs[0]])

x = tf.random.uniform((1, 128, 128, 3))

with tf.GradientTape() as tape:
    tape.watch(x)
    conv, pred = gm(x, training=False)
    score = tf.reduce_sum(pred)

grad_x, grad_conv = tape.gradient(score, [x, conv])

print("Prediction:", pred.numpy())
print("Input gradient available:", grad_x is not None)
print("Feature-map gradient available:", grad_conv is not None)

if grad_x is not None:
    print("Input gradient mean:", float(tf.reduce_mean(tf.abs(grad_x)).numpy()))

if grad_conv is not None:
    print("Feature-map gradient mean:", float(tf.reduce_mean(tf.abs(grad_conv)).numpy()))
