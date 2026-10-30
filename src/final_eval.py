"""Final comparison: each model's validation-selected configuration
(results/tuning/<model>.json) trained with 5 seeds and scored once on the
test set. Usage: python src/final_eval.py [MODEL ...]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from protocol import run, data, metrics
from tune import GRID

SEEDS = [0, 1, 2, 3, 4]


def main(which):
    os.makedirs("results/final", exist_ok=True)
    for name in which:
        tun = json.load(open(f"results/tuning/{name}.json"))
        cfg = tun["rows"][tun["best"]]["cfg"]
        runs, preds, extra = [], [], {}
        for s in SEEDS:
            out = run(name, cfg, s, attention=(name == "Transformer" and s == 0))
            v, t, p, hist, info = out[:5]
            if s == 0:
                extra["history"] = hist
                if name == "Transformer":  # attention of hour t over the window, last block
                    X = data(cfg.get("window", 24))[0]["test"][0]
                    att = out[5].predict(X, verbose=0)[-1]           # (n, heads, q, k)
                    extra["attention_last_query"] = att[:, :, -1, :].mean(axis=(0, 1)).tolist()
            runs.append(dict(val_MAE=v["MAE"], **t, **info))
            preds.append(p.tolist())
            print(name, s, {k: round(x, 4) for k, x in runs[-1].items()}, flush=True)
        summary = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))]
                   for k in runs[0]}
        json.dump(dict(model=name, cfg=cfg, seeds=SEEDS, runs=runs, summary=summary,
                       preds=preds, **extra),
                  open(f"results/final/{name}.json", "w"), indent=1)

    # shared reference data (identical for every model thanks to ALIGN)
    d, meta = data(24)
    yte = d["test"][1]*meta["y_sd"] + meta["y_mu"]
    ps = meta["persistence"]["test"]
    json.dump(dict(y_test=yte.tolist(), persistence_test=ps.tolist(),
                   persistence=metrics(yte, ps),
                   target_time=[str(t) for t in meta["target_time"]["test"]],
                   n={k: len(v[1]) for k, v in d.items()}),
              open("results/final/reference.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or list(GRID))
