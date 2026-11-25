import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# Overlap Patch Embedding
# ============================================================

@keras.utils.register_keras_serializable(package="DamageClassifier")
class OverlapPatchEmbedding(layers.Layer):

    def __init__(self, embed_dim, patch_size, stride, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.stride = stride

        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="same"
        )

        self.norm = layers.LayerNormalization(
            epsilon=1e-6
        )

    def call(self, x):
        # Convolution
        x = self.projection(x)

        # Get shape
        shape = tf.shape(x)

        batch = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # Convert image feature map to sequence
        x = tf.reshape(
            x,
            [batch, height * width, channels]
        )

        # Normalization
        x = self.norm(x)

        return x, height, width

    def get_config(self):
        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "stride": self.stride
        })

        return config


# ============================================================
# Mix Feed Forward Network
# ============================================================

@keras.utils.register_keras_serializable(package="DamageClassifier")
class MixFFN(layers.Layer):

    def __init__(
        self,
        embed_dim,
        expansion=4,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.expansion = expansion
        self.dropout = dropout

        hidden_dim = embed_dim * expansion

        self.dense1 = layers.Dense(
            hidden_dim,
            activation=tf.nn.gelu
        )

        self.dropout1 = layers.Dropout(
            dropout
        )

        self.dense2 = layers.Dense(
            embed_dim
        )

        self.dropout2 = layers.Dropout(
            dropout
        )

    def call(self, x, training=False):

        x = self.dense1(x)

        x = self.dropout1(
            x,
            training=training
        )

        x = self.dense2(x)

        x = self.dropout2(
            x,
            training=training
        )

        return x

    def get_config(self):
        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "expansion": self.expansion,
            "dropout": self.dropout
        })

        return config


# ============================================================
# SegFormer Block
# ============================================================

@keras.utils.register_keras_serializable(package="DamageClassifier")
class SegFormerBlock(layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # First normalization
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6
        )

        # Multi-head self attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )

        # Second normalization
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6
        )

        # Feed Forward Network
        self.mlp = MixFFN(
            embed_dim,
            dropout=dropout
        )

    def call(self, x, training=False):

        # -----------------------------
        # Self Attention
        # -----------------------------

        attention_input = self.norm1(x)

        attention_output = self.attention(
            attention_input,
            attention_input,
            training=training
        )

        # Residual connection
        x = x + attention_output

        # -----------------------------
        # Feed Forward Network
        # -----------------------------

        mlp_input = self.norm2(x)

        mlp_output = self.mlp(
            mlp_input,
            training=training
        )

        # Residual connection
        x = x + mlp_output

        return x

    def get_config(self):
        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout
        })

        return config