"""Common Phase 3 protocol shared by tune.py and final_eval.py.

All runs use the same target hours (windows aligned to ALIGN = 48 h, the
longest window searched), the same chronological split, Adam, MSE loss on the
standardised target, batch 64, up to 100 epochs with early stopping on
validation loss (patience 10, best weights restored)."""
import os, sys, time
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import build
from models import MODELS

ALIGN = 48
EPOCHS, BATCH, PATIENCE = 100, 64, 10
TRAIN_KEYS = ("window", "lr", "clipnorm", "warmup", "reduce_lr")

if os.environ.get("TF_THREADS"):
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_THREADS"]))
    tf.config.threading.set_inter_op_parallelism_threads(1)

_cache = {}


def data(window):
    if window not in _cache:
        _cache[window] = build(window=window, align=ALIGN)
    return _cache[window]


def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)))


def run(name, cfg, seed, attention=False):
    """Train one model/config/seed. Returns (val metrics, test metrics,
    test predictions in mg/m3, history, info[, attention model])."""
    cfg = dict(cfg)
    window = cfg.get("window", 24)
    lr, clip = cfg.get("lr", 1e-3), cfg.get("clipnorm")
    warmup, reduce_lr = cfg.get("warmup", 0), cfg.get("reduce_lr", False)
    kw = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in cfg.items() if k not in TRAIN_KEYS}
    d, meta = data(window)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = d["train"], d["val"], d["test"]
    mu, sd = meta["y_mu"], meta["y_sd"]

    tf.keras.utils.set_random_seed(seed)
    if attention:
        m, att = MODELS[name](window, Xtr.shape[2], with_attention=True, **kw)
    else:
        m, att = MODELS[name](window, Xtr.shape[2], **kw), None
    opt = tf.keras.optimizers.Adam(lr, clipnorm=clip) if clip else tf.keras.optimizers.Adam(lr)
    m.compile(opt, loss="mse")
    cbs = [tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)]
    if warmup:  # linear learning-rate warm-up over the first `warmup` epochs
        cbs.append(tf.keras.callbacks.LearningRateScheduler(
            lambda e: lr*min(1.0, (e+1)/warmup)))
    if reduce_lr:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-5))
    t0 = time.time()
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS, batch_size=BATCH,
              verbose=0, callbacks=cbs)
    secs = time.time() - t0
    pva = m.predict(Xva, verbose=0).ravel()*sd + mu
    pte = m.predict(Xte, verbose=0).ravel()*sd + mu
    info = dict(epochs=len(h.history["loss"]), secs=secs, params=int(m.count_params()))
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items() if k in ("loss", "val_loss")}
    out = (metrics(yva*sd + mu, pva), metrics(yte*sd + mu, pte), pte, hist, info)
    return out + (att,) if attention else out
