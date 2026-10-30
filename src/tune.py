"""Validation-based hyperparameter search with an equal budget per model:
6 configurations x 2 seeds each, ranked by mean VALIDATION MAE (mg/m3).
The test set is never used for selection (test metrics are not even computed
into the ranking). Usage: python src/tune.py [MODEL ...]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from protocol import run

SEEDS = [0, 1]
# config 0 of every model is the interim (Phase 2) architecture
GRID = {
    "MLP": [  # Ashwin Singh Kushwaha
        dict(window=24),
        dict(window=12),
        dict(window=48),
        dict(window=24, units=[256, 128]),
        dict(window=24, dropout=0.3, l2=1e-4),
        dict(window=48, dropout=0.3, l2=1e-4),
    ],
    "1D-CNN": [  # Mrittika Sarkar
        dict(window=24),                                 # k3, d1-2-4, RF 15 h
        dict(window=24, dilations=[1, 2, 4, 8]),         # RF 31 h (covers window)
        dict(window=24, kernel=5),                       # RF 29 h
        dict(window=24, dilations=[1, 1, 1]),            # non-dilated, RF 7 h
        dict(window=48, dilations=[1, 2, 4, 8]),         # RF 31 h on 48 h input
        dict(window=24, filters=32),
    ],
    "LSTM": [  # Sagnik Chattopadhyay
        dict(window=24),
        dict(window=24, clipnorm=1.0),
        dict(window=24, clipnorm=1.0, reduce_lr=True),
        dict(window=24, clipnorm=1.0, units=[32]),
        dict(window=12, clipnorm=1.0),
        dict(window=48, clipnorm=1.0),
    ],
    "Transformer": [  # Tanish Sam Mathai
        dict(window=24),                                 # d32, 4 heads
        dict(window=24, d=16, heads=2, ff=32),
        dict(window=24, d=64, heads=4, ff=128),
        dict(window=24, dropout=0.2),
        dict(window=24, warmup=5),
        dict(window=48),
    ],
}


def main(which):
    os.makedirs("results/tuning", exist_ok=True)
    for name in which:
        rows = []
        for i, cfg in enumerate(GRID[name]):
            val = []
            for s in SEEDS:
                v, _, _, _, info = run(name, cfg, s)
                val.append(v["MAE"])
                print(name, i, cfg, s, round(v["MAE"], 4), info, flush=True)
            rows.append(dict(cfg=cfg, val_mae=[float(np.mean(val)), float(np.std(val))],
                             params=info["params"]))
        best = min(range(len(rows)), key=lambda i: rows[i]["val_mae"][0])
        json.dump(dict(model=name, seeds=SEEDS, rows=rows, best=best),
                  open(f"results/tuning/{name}.json", "w"), indent=1)
        print(name, "best config", best, rows[best], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(GRID))
