"""Four architecture families, all mapping (window, n_features) -> 1 value."""
import tensorflow as tf
from tensorflow.keras import layers, Model


def mlp(window, nf):            # Member 1 - classical neural baseline
    x_in = layers.Input((window, nf))
    x = layers.Flatten()(x_in)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    
    return Model(x_in, layers.Dense(1)(x), name="MLP")


def cnn1d(window, nf, readout="last"):  # Member 2 - local temporal patterns
    """readout='gap' (v1): average over all 24 steps.
    readout='last' (v2, selected on validation): features at hour t only;
    with dilations 1,2,4 and kernel 3 its causal receptive field is 15 h."""
    x_in = layers.Input((window, nf))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x_in)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu", dilation_rate=2)(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu", dilation_rate=4)(x)
    if readout == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    return Model(x_in, layers.Dense(1)(x), name="CNN1D")


def lstm(window, nf):           # Member 3 - recurrent memory
    x_in = layers.Input((window, nf))
    x = layers.LSTM(64, return_sequences=True)(x_in)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    return Model(x_in, layers.Dense(1)(x), name="LSTM")


class PositionalEmbedding(layers.Layer):
    def __init__(self, window, d, **kw):
        super().__init__(**kw)
        self.window = window
        self.pos = layers.Embedding(window, d)

    def call(self, x):
        return x + self.pos(tf.range(self.window))


def transformer(window, nf, d=32, heads=4, blocks=2, ff=64):  # Member 4
    x_in = layers.Input((window, nf))
    x = layers.Dense(d)(x_in)
    x = PositionalEmbedding(window, d)(x)
    for _ in range(blocks):
        a = layers.MultiHeadAttention(heads, d // heads, dropout=0.1)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff, activation="relu")(x)
        f = layers.Dense(d)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(0.1)(f))
    x = layers.Lambda(lambda z: z[:, -1, :])(x)   # representation of hour t
    x = layers.Dense(32, activation="relu")(x)
    return Model(x_in, layers.Dense(1)(x), name="Transformer")


MODELS = {"MLP": mlp, "1D-CNN": cnn1d, "LSTM": lstm, "Transformer": transformer}
# Reviewed and verified 
