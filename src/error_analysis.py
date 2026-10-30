"""Final comparison table, error analysis for every model and Phase 3 figures.
Reads results/tuning/*.json and results/final/*.json (from tune.py and
final_eval.py); writes results/final_summary.json and figures/fig7..fig14."""
import json, math, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .3, "savefig.dpi": 200, "savefig.bbox": "tight"})
C = {"MLP": "#4C78A8", "1D-CNN": "#F58518", "LSTM": "#54A24B", "Transformer": "#B279A2"}
models = list(C)
os.makedirs("figures", exist_ok=True)

ref = json.load(open("results/final/reference.json"))
F = {k: json.load(open(f"results/final/{k}.json")) for k in models}
T = {k: json.load(open(f"results/tuning/{k}.json")) for k in models}
y = np.array(ref["y_test"]); ps = np.array(ref["persistence_test"])
tt = pd.to_datetime(ref["target_time"])
P = {k: np.array(F[k]["preds"]) for k in models}        # (seeds, n)
ens = {k: P[k].mean(0) for k in models}                  # 5-seed mean prediction
out = {"persistence": ref["persistence"], "n": ref["n"]}


def mae(p, mask=slice(None)):
    return float(np.mean(np.abs(p - y)[..., mask]))


def grouped(keys, labels):
    """MAE per group: mean +- std over seeds for each model, plus persistence."""
    g = {"labels": labels, "count": [int((keys == i).sum()) for i in range(len(labels))],
         "persistence": [mae(ps, keys == i) for i in range(len(labels))]}
    for k in models:
        per_seed = np.array([[mae(p, keys == i) for i in range(len(labels))] for p in P[k]])
        g[k] = [per_seed.mean(0).tolist(), per_seed.std(0).tolist()]
    return g


# 1. comparison table and tuning outcome
out["table"] = {k: dict(F[k]["summary"], cfg=F[k]["cfg"],
                        interim_cfg_val_mae=T[k]["rows"][0]["val_mae"],
                        best_cfg_val_mae=T[k]["rows"][T[k]["best"]]["val_mae"],
                        best_cfg_index=T[k]["best"]) for k in models}
best = min(models, key=lambda k: F[k]["summary"]["MAE"][0])
out["best"] = best

# 2. error by true CO level, bias at peaks
bins = [1, 2, 3, 4]; lab = ["<1", "1–2", "2–3", "3–4", "≥4"]
lvl = np.digitize(y, bins)
out["by_level"] = grouped(lvl, lab)
out["bias_by_level"] = {k: [float(np.mean((P[k] - y)[:, lvl == i])) for i in range(5)] for k in models}
out["bias_by_level"]["persistence"] = [float(np.mean((ps - y)[lvl == i])) for i in range(5)]

# 3. hour of day (target hour), 4. weekday vs weekend, 5. month (drift)
out["by_hour"] = grouped(tt.hour.values, [f"{h:02d}" for h in range(24)])
out["by_daytype"] = grouped((tt.dayofweek.values >= 5).astype(int), ["Weekday", "Weekend"])
months = sorted(set(tt.to_period("M")))
out["by_month"] = grouped(np.array([months.index(m) for m in tt.to_period("M")]),
                          [m.strftime("%b %Y") for m in months])
out["val_vs_test_mae"] = {k: [F[k]["summary"]["val_MAE"][0], F[k]["summary"]["MAE"][0]] for k in models}


# 6. Diebold-Mariano test on squared errors of the 5-seed mean predictions
#    (Newey-West variance, lag n^(1/3), because hourly errors are autocorrelated)
def dm(e1, e2):
    d = e1**2 - e2**2; n = len(d); L = int(round(n**(1/3))); dc = d - d.mean()
    v = dc @ dc / n + 2*sum((1 - l/(L+1)) * (dc[l:] @ dc[:-l]) / n for l in range(1, L+1))
    stat = d.mean() / math.sqrt(v/n)
    return dict(DM=float(stat), p=float(math.erfc(abs(stat)/math.sqrt(2))))


out["ensemble"] = {k: dict(MAE=mae(ens[k]), RMSE=float(np.sqrt(np.mean((ens[k]-y)**2)))) for k in models}
out["dm_vs_best"] = {k: dm(ens[k]-y, ens[best]-y) for k in models if k != best}
out["dm_vs_persistence"] = {k: dm(ens[k]-y, ps-y) for k in models}
if "attention_last_query" in F["Transformer"]:
    out["attention"] = F["Transformer"]["attention_last_query"]
json.dump(out, open("results/final_summary.json", "w"), indent=1)

# Fig 1 (final): CO(GT) series with the split, Phase 3 sample counts
import sys; sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load
raw = load(); n = len(raw); cut = [0, int(n*.70), int(n*.85), n]
fig, ax = plt.subplots(figsize=(9, 3))
co = raw["CO_GT"].resample("D").mean()
ax.plot(co.index, co.values, color="#333", lw=1)
for i, (k, col) in enumerate([("train", "#4C78A8"), ("val", "#F58518"), ("test", "#54A24B")]):
    ax.axvspan(raw.index[cut[i]], raw.index[cut[i+1]-1], color=col, alpha=.18,
               label=f"{k.capitalize()} ({ref['n'][k]:,} samples)")
ax.set_ylabel("Daily mean CO(GT), mg/m³"); ax.legend(loc="upper left", fontsize=8)
ax.set_title("CO(GT) over time with chronological train / validation / test split")
fig.savefig("figures/fig1_split_final.png"); plt.close(fig)

# Fig 7: tuning (validation MAE per configuration, 2 seeds)
fig, axs = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for ax, k in zip(axs, models):
    rows = T[k]["rows"]; m = [r["val_mae"][0] for r in rows]; s = [r["val_mae"][1] for r in rows]
    cols = [C[k] if i == T[k]["best"] else "#bbb" for i in range(len(rows))]
    ax.bar(range(len(rows)), m, yerr=s, capsize=3, color=cols)
    ax.set_xticks(range(len(rows)), [f"c{i}" for i in range(len(rows))])
    ax.set_title(k, fontsize=9); ax.set_xlabel("Configuration (c0 = interim)")
axs[0].set_ylabel("Validation MAE, mg/m³"); axs[0].set_ylim(0.4, None)
fig.suptitle("Validation-based search, equal budget (6 configs × 2 seeds); selected config coloured", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig7_tuning.png"); plt.close(fig)

# Fig 8: final comparison (5 seeds)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, met in zip(axs, ["MAE", "RMSE", "R2"]):
    m = [F[k]["summary"][met][0] for k in models]; s = [F[k]["summary"][met][1] for k in models]
    ax.bar(models, m, yerr=s, capsize=4, color=[C[k] for k in models])
    ax.axhline(ref["persistence"][met], ls="--", color="k", lw=1, label="Persistence")
    for i, v in enumerate(m):
        ax.text(i, v/2, f"{v:.3f}", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_title(met + (" (higher is better)" if met == "R2" else " (lower is better)"), fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
axs[0].legend(fontsize=7)
fig.suptitle("Final test-set comparison, tuned models (mean ± std over 5 seeds)", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig8_final_comparison.png"); plt.close(fig)


def grouped_bars(g, fname, title, xlabel, figsize=(8, 3.3)):
    fig, ax = plt.subplots(figsize=figsize); x = np.arange(len(g["labels"])); w = .16
    ax.bar(x - 2*w, g["persistence"], w, color="#999", label="Persistence")
    for i, k in enumerate(models):
        ax.bar(x + (i-1)*w, g[k][0], w, yerr=g[k][1], capsize=2, color=C[k], label=k)
    ax.set_xticks(x, [f"{l}\n(n={c})" for l, c in zip(g["labels"], g["count"])])
    ax.set_xlabel(xlabel); ax.set_ylabel("MAE, mg/m³"); ax.legend(fontsize=7, ncol=5)
    ax.set_title(title); fig.savefig(fname); plt.close(fig)


grouped_bars(out["by_level"], "figures/fig9_error_by_level.png",
             "Error by true CO level, all models (mean ± std over 5 seeds)", "True CO(GT) range, mg/m³")
grouped_bars(out["by_month"], "figures/fig11_error_by_month.png",
             "Test error by month (sensor drift check)", "Month of target hour", figsize=(6.5, 3.2))

# Fig 10: error by hour of day + mean CO profile
g = out["by_hour"]; h = np.arange(24)
fig, ax = plt.subplots(figsize=(9, 3.3))
ax.plot(h, g["persistence"], "--", color="#666", label="Persistence")
for k in models:
    ax.plot(h, g[k][0], color=C[k], label=k, marker="o", ms=3)
ax2 = ax.twinx(); prof = [y[tt.hour.values == i].mean() for i in h]
ax2.fill_between(h, prof, color="#ccc", alpha=.35, step="mid"); ax2.set_ylabel("Mean true CO, mg/m³", color="#777")
ax2.grid(False); ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
ax.set_xticks(h[::2]); ax.set_xlabel("Hour of day of target"); ax.set_ylabel("MAE, mg/m³")
ax.legend(fontsize=7, ncol=5, loc="upper left"); ax.set_title("Test error by hour of day (shaded: mean CO profile)")
fig.savefig("figures/fig10_error_by_hour.png"); plt.close(fig)

# Fig 12: Transformer attention from hour t over the input window
if "attention" in out:
    a = np.array(out["attention"]); lags = np.arange(len(a)-1, -1, -1)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(-lags, a, color=C["Transformer"])
    ax.axhline(1/len(a), ls="--", color="k", lw=1, label="Uniform (1/window)"); ax.legend(fontsize=7)
    ax.set_xlabel("Input hour relative to t (0 = most recent)"); ax.set_ylabel("Mean attention weight")
    ax.set_title("Transformer: last-block attention of hour t (test set, averaged over heads)")
    fig.savefig("figures/fig12_attention.png"); plt.close(fig)

# Fig 13: predictions vs actual around the largest peak in the test set
c = int(np.argmax(y)); lo = max(0, c-96); hi = min(len(y), lo+168)
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(range(lo, hi), y[lo:hi], color="k", lw=1.6, label="Actual CO(GT)")
ax.plot(range(lo, hi), ps[lo:hi], ":", color="#666", lw=1, label="Persistence")
for k in models:
    ax.plot(range(lo, hi), ens[k][lo:hi], lw=1, color=C[k], label=k, alpha=.9)
ax.set_xlabel(f"Test sample index ({tt[lo]:%d %b} – {tt[hi-1]:%d %b %Y})"); ax.set_ylabel("CO(GT), mg/m³")
ax.legend(ncol=6, fontsize=7, loc="upper left")
ax.set_title("Tuned models (5-seed mean prediction) around the largest test-set peak")
fig.savefig("figures/fig13_peak_predictions.png"); plt.close(fig)

# Fig 14: predicted vs actual scatter for each model
fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for ax, k in zip(axs, models):
    ax.scatter(y, ens[k], s=3, alpha=.35, color=C[k]); ax.plot([0, 12], [0, 12], "k--", lw=.8)
    ax.set_title(k, fontsize=9); ax.set_xlabel("Actual, mg/m³"); ax.set_xlim(0, 12); ax.set_ylim(0, 12)
axs[0].set_ylabel("Predicted, mg/m³")
fig.suptitle("Predicted vs actual (test set): all models under-predict high concentrations", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig14_scatter.png"); plt.close(fig)

print(json.dumps({k: {m: round(v[0], 4) for m, v in F[k]["summary"].items()} for k in models}, indent=1))
print("best =", best, "| DM vs best:", out["dm_vs_best"])
