"""Four architecture families, all mapping (window, n_features) -> 1 value.

Default arguments reproduce the interim (Phase 2) models; the extra
arguments are the hyperparameters searched in src/tune.py."""
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


def mlp(window, nf, units=(128, 64), dropout=0.2, l2=0.0):  # Member 1 - classical neural baseline
    reg = regularizers.l2(l2) if l2 else None
    x_in = layers.Input((window, nf))
    x = layers.Flatten()(x_in)
    x = layers.Dense(units[0], activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units[1], activation="relu", kernel_regularizer=reg)(x)
    return Model(x_in, layers.Dense(1)(x), name="MLP")


def cnn1d(window, nf, readout="last", filters=64, kernel=3, dilations=(1, 2, 4),
          dropout=0.2):  # Member 2 - local temporal patterns
    """readout='gap' (v1): average over all steps.
    readout='last' (v2, selected on validation): features at hour t only.
    Causal receptive field = 1 + (kernel-1)*sum(dilations); 15 h by default."""
    x_in = layers.Input((window, nf))
    x = x_in
    for d in dilations:
        x = layers.Conv1D(filters, kernel, padding="causal", activation="relu",
                          dilation_rate=d)(x)
    if readout == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    return Model(x_in, layers.Dense(1)(x), name="CNN1D")


def lstm(window, nf, units=(64, 32), dropout=0.2):  # Member 3 - recurrent memory
    x_in = layers.Input((window, nf))
    x = x_in
    for i, u in enumerate(units):
        x = layers.LSTM(u, return_sequences=i < len(units)-1)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    return Model(x_in, layers.Dense(1)(x), name="LSTM")


class PositionalEmbedding(layers.Layer):
    def __init__(self, window, d, **kw):
        super().__init__(**kw)
        self.window = window
        self.pos = layers.Embedding(window, d)

    def call(self, x):
        return x + self.pos(tf.range(self.window))


def transformer(window, nf, d=32, heads=4, blocks=2, ff=64, dropout=0.1,
                with_attention=False):  # Member 4
    """with_attention=True also returns a model that outputs the attention
    weights of every block (shares weights with the forecasting model)."""
    x_in = layers.Input((window, nf))
    x = layers.Dense(d)(x_in)
    x = PositionalEmbedding(window, d)(x)
    scores = []
    for _ in range(blocks):
        a, s = layers.MultiHeadAttention(heads, d // heads, dropout=dropout)(
            x, x, return_attention_scores=True)
        scores.append(s)
        x = layers.LayerNormalization(epsilon=1e-6)(x + a)
        f = layers.Dense(ff, activation="relu")(x)
        f = layers.Dense(d)(f)
        x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dropout)(f))
    x = layers.Lambda(lambda z: z[:, -1, :])(x)   # representation of hour t
    x = layers.Dense(32, activation="relu")(x)
    m = Model(x_in, layers.Dense(1)(x), name="Transformer")
    return (m, Model(x_in, scores)) if with_attention else m


MODELS = {"MLP": mlp, "1D-CNN": cnn1d, "LSTM": lstm, "Transformer": transformer}
