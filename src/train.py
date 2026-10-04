"""Train and evaluate all models under one protocol (3 seeds each)."""
import json, time, os, sys
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import build, WINDOW
from models import MODELS

SEEDS = [0, 1, 2]
EPOCHS, BATCH, LR, PATIENCE = 100, 64, 1e-3, 10


def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)))


def main(which=None):
    data, meta = build()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    mu, sd = meta["y_mu"], meta["y_sd"]
    yte_mg = yte*sd + mu
    res = {"persistence": metrics(yte_mg, meta["persistence"]["test"])}
    preds, hist = {}, {}
    for name, fn in MODELS.items():
        if which and name not in which:
            continue
        runs = []
        for s in SEEDS:
            tf.keras.utils.set_random_seed(s)
            m = fn(WINDOW, Xtr.shape[2])
            m.compile(tf.keras.optimizers.Adam(LR), loss="mse")
            t0 = time.time()
            h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                      batch_size=BATCH, verbose=0,
                      callbacks=[tf.keras.callbacks.EarlyStopping(
                          patience=PATIENCE, restore_best_weights=True)])
            p = m.predict(Xte, verbose=0).ravel()*sd + mu
            r = metrics(yte_mg, p)
            r.update(epochs=len(h.history["loss"]), secs=time.time()-t0,
                     params=int(m.count_params()))
            runs.append(r)
            if s == 0:
                preds[name] = p.tolist()
                hist[name] = {k: [float(v) for v in vals] for k, vals in h.history.items()}
            print(name, s, {k: round(v, 4) for k, v in r.items()}, flush=True)
        res[name] = {k: [float(np.mean([r[k] for r in runs])),
                         float(np.std([r[k] for r in runs]))] for k in runs[0]}
    out = dict(results=res, preds=preds, history=hist, y_test=yte_mg.tolist(),
               persistence_test=meta["persistence"]["test"].tolist(),
               n=dict(train=len(ytr), val=len(yva), test=len(yte)),
               split_dates={k: [str(a), str(b)] for k, (a, b) in meta["split_dates"].items()},
               split_rows=meta["split_rows"], features=meta["features"],
               missing_pct=meta["missing_pct"])
    json.dump(out, open("results/results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
# Checked learning rate stability 
