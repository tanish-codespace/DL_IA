"""Chronological preprocessing pipeline for next-hour CO(GT) forecasting.

Target: CO(GT) at hour t+1. Inputs: a window of the WINDOW most recent hours
(t-WINDOW+1 ... t). Every statistic (imputation medians, scaler) is fitted on
the training period only, and the split is strictly chronological.
"""
import numpy as np
import pandas as pd

WINDOW = 24                      # hours of history per sample
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 0.15 -> test
TARGET = "CO_GT"
DROP = ["NMHC_GT"]               # ~90% missing -> excluded
MAX_FFILL = 3                    # carry a value forward at most 3 hours


def load(path="data/AirQualityUCI.csv"):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    format="%d-%m-%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"] + DROP).set_index("datetime").sort_index()
    df = df.replace(-200, np.nan)  # dataset's missing-value code
    return df


def add_time_features(df):
    h, dow = df.index.hour, df.index.dayofweek
    df["hour_sin"], df["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    df["dow_sin"], df["dow_cos"] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    return df


def build(path="data/AirQualityUCI.csv", window=WINDOW):
    raw = load(path)
    n = len(raw)
    i_tr, i_va = int(n*TRAIN_FRAC), int(n*(TRAIN_FRAC+VAL_FRAC))
    sensors = list(raw.columns)

    # missing-value indicators (lets models know a value was imputed)
    target_observed = raw[TARGET].notna().values
    feat = raw.copy()
    feat["CO_missing"] = (~raw[TARGET].notna()).astype(float)

    # causal imputation: short forward fill (past only), then TRAIN median
    feat[sensors] = feat[sensors].ffill(limit=MAX_FFILL)
    med = feat.iloc[:i_tr][sensors].median()
    feat[sensors] = feat[sensors].fillna(med)
    feat = add_time_features(feat)

    # standardise with TRAIN statistics only
    mu, sd = feat.iloc[:i_tr].mean(), feat.iloc[:i_tr].std().replace(0, 1)
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "CO_missing"]:
        mu[c], sd[c] = 0.0, 1.0
    X_all = ((feat - mu) / sd).values.astype("float32")
    y_all = raw[TARGET].values.astype("float32")  # unscaled, NaN where missing
    y_mu, y_sd = float(mu[TARGET]), float(sd[TARGET])

    # windows: inputs end at t, target at t+1; keep only OBSERVED targets
    splits = {"train": ([], []), "val": ([], []), "test": ([], [])}
    persist = {"train": [], "val": [], "test": []}
    co_col = list(feat.columns).index(TARGET)
    for t in range(window-1, n-1):
        tgt = t + 1
        if not target_observed[tgt]:
            continue
        # a sample belongs to the split containing its target, and its whole
        # window must lie inside that split (no boundary leakage)
        if tgt < i_tr:
            s, start = "train", 0
        elif tgt < i_va:
            s, start = "val", i_tr
        else:
            s, start = "test", i_va
        if t-window+1 < start:
            continue
        splits[s][0].append(X_all[t-window+1:t+1])
        splits[s][1].append((y_all[tgt]-y_mu)/y_sd)
        persist[s].append(X_all[t, co_col]*y_sd + y_mu)  # naive reference

    out = {k: (np.stack(v[0]), np.array(v[1], "float32")) for k, v in splits.items()}
    meta = dict(features=list(feat.columns), y_mu=y_mu, y_sd=y_sd,
                n_rows=n, split_rows=(i_tr, i_va-i_tr, n-i_va),
                split_dates={"train": (raw.index[0], raw.index[i_tr-1]),
                             "val": (raw.index[i_tr], raw.index[i_va-1]),
                             "test": (raw.index[i_va], raw.index[-1])},
                persistence={k: np.array(v) for k, v in persist.items()},
                missing_pct=(raw.isna().mean()*100).round(1).to_dict())
    return out, meta


if __name__ == "__main__":
    d, m = build()
    for k, (X, y) in d.items():
        print(k, X.shape, y.shape, m["split_dates"][k])
    print(m["features"])
