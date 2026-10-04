"""Generate report figures from results/results.json."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .3, "savefig.dpi": 200, "savefig.bbox": "tight"})
R = json.load(open("results/results.json"))
C = {"MLP": "#4C78A8", "1D-CNN": "#F58518", "LSTM": "#54A24B", "Transformer": "#B279A2"}
os.makedirs("figures", exist_ok=True)

# Fig 1: CO(GT) series with chronological split shading
raw = load()
fig, ax = plt.subplots(figsize=(9, 3))
co = raw["CO_GT"].resample("D").mean()
ax.plot(co.index, co.values, color="#333", lw=1)
cols = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
for k, (a, b) in R["split_dates"].items():
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=cols[k], alpha=.18,
               label=f"{k.capitalize()} ({R['n'][k]:,} samples)")
ax.set_ylabel("Daily mean CO(GT), mg/m³"); ax.legend(loc="upper left", fontsize=8)
ax.set_title("CO(GT) over time with chronological train / validation / test split")
fig.savefig("figures/fig1_split.png"); plt.close(fig)

# Fig 2: missingness per variable
_r = pd.read_csv("data/AirQualityUCI.csv").drop(columns=["Date", "Time"]).replace(-200, np.nan)
mp = (_r.isna().mean()*100).sort_values()
fig, ax = plt.subplots(figsize=(7, 3.4))
bars = ax.barh(mp.index, mp.values, color=["#E45756" if v > 50 else "#4C78A8" for v in mp.values])
for b, v in zip(bars, mp.values):
    ax.text(v + 1, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Missing values (% of 9,357 hours, code -200)"); ax.set_xlim(0, 100)
ax.set_title("Missing data by variable (NMHC(GT) dropped)")
fig.savefig("figures/fig2_missing.png"); plt.close(fig)

# Fig 3: model comparison (mean ± std over 3 seeds) + persistence line
res = R["results"]; models = list(C)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, met in zip(axs, ["MAE", "RMSE", "R2"]):
    m = [res[k][met][0] for k in models]; s = [res[k][met][1] for k in models]
    ax.bar(models, m, yerr=s, capsize=4, color=[C[k] for k in models])
    ax.axhline(res["persistence"][met], ls="--", color="k", lw=1, label="Persistence")
    for i, v in enumerate(m):
        ax.text(i, v/2, f"{v:.3f}", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_title(met + (" (higher is better)" if met == "R2" else " (lower is better)"), fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
axs[0].legend(fontsize=7)
fig.suptitle("Test-set comparison (mean ± std over 3 seeds)", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig3_comparison.png"); plt.close(fig)

# Fig 4: predicted vs actual, first 168 test samples
y = np.array(R["y_test"]); n = 168
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(y[:n], color="k", lw=1.6, label="Actual CO(GT)")
for k in ["MLP", "1D-CNN", "LSTM", "Transformer"]:
    ax.plot(np.array(R["preds"][k])[:n], lw=1, color=C[k], label=k, alpha=.9)
ax.set_xlabel("Test sample index (observed hours from 5 Feb 2005)")
ax.set_ylabel("CO(GT), mg/m³"); ax.legend(ncol=5, fontsize=8, loc="upper right")
ax.set_title("Next-hour predictions vs. actual (first 168 test samples, seed 0)")
fig.savefig("figures/fig4_predictions.png"); plt.close(fig)

# Fig 5: training / validation loss
fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
for ax, k in zip(axs, models):
    h = R["history"][k]
    ax.plot(h["loss"], color=C[k], label="Train"); ax.plot(h["val_loss"], "--", color="k", label="Val")
    ax.set_title(k, fontsize=9); ax.set_xlabel("Epoch")
axs[0].set_ylabel("MSE (standardised)"); axs[0].legend(fontsize=7); axs[0].set_ylim(0, 1)
fig.suptitle("Training and validation loss (seed 0, early stopping patience 10)", fontsize=10)
fig.tight_layout(); fig.savefig("figures/fig5_loss.png"); plt.close(fig)

# Fig 6: error analysis - MAE by true-CO bin, best model vs persistence
best = min(models, key=lambda k: res[k]["MAE"][0])
p = np.array(R["preds"][best]); ps = np.array(R["persistence_test"])
bins = [0, 1, 2, 3, 4, 12]; lab = ["<1", "1–2", "2–3", "3–4", "≥4"]
idx = np.digitize(y, bins[1:-1])
mae_b = [np.mean(np.abs(p-y)[idx == i]) for i in range(5)]
mae_p = [np.mean(np.abs(ps-y)[idx == i]) for i in range(5)]
cnt = [int((idx == i).sum()) for i in range(5)]
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(5)
ax.bar(x-.2, mae_p, .4, color="#999", label="Persistence")
ax.bar(x+.2, mae_b, .4, color=C[best], label=f"{best} (seed 0)")
ax.set_xticks(x, [f"{l}\n(n={c})" for l, c in zip(lab, cnt)])
ax.set_xlabel("True CO(GT) range, mg/m³"); ax.set_ylabel("MAE, mg/m³"); ax.legend(fontsize=8)
ax.set_title("Error analysis: absolute error grows with CO concentration")
fig.savefig("figures/fig6_error_bins.png"); plt.close(fig)
json.dump({"best": best, "bins": lab, "count": cnt, "mae_best": mae_b, "mae_persist": mae_p},
          open("results/error_bins.json", "w"), indent=1)
print("figures done; best =", best)
