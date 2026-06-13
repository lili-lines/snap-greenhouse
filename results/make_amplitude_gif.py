"""
Variant of the drift demo where the SHAPE of the daily cycle changes (its
amplitude doubles) instead of a simple level shift — so the frozen batch model
is visibly the *wrong shape*, not just flat.

Controlled schematic:
- level = real slow trend of the greenhouse temperature (rolling 48 h mean);
- on top, a CLEAN daily cycle whose **amplitude doubles** at the shift
  (e.g. a season change / cover removed: the greenhouse now swings much more);
- features = time-of-day only (hour sin/cos), so the change is not visible in
  the inputs and can only be followed by *learning*;
- two identical River models warm up together; at the shift one keeps learning
  (online), the other is frozen (batch).

Output : results/figures/online_adapts_amplitude.gif
Run    : python results/make_amplitude_gif.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from river import linear_model, preprocessing, optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import cfg

NB = ROOT / "notebooks"

# ============================== config ==============================
SEG_START  = 3000    # start of the real slice used for the baseline trend
N_HOURS    = 600
SHIFT_AT   = 300     # step where the amplitude change happens
AMP_BEFORE = 5.0     # daily-cycle amplitude before the shift (degC)
AMP_AFTER  = 11.0    # daily-cycle amplitude after the shift (degC)
NOISE      = 0.7     # observation noise (degC)
LR         = 0.1     # river Adam learning rate
ROLL       = 24
OUT_PATH   = ROOT / "results" / "figures" / "online_adapts_amplitude.gif"
FPS        = 12
DPI        = 90
# ====================================================================


def _features(ts):
    h = ts.hour
    return {"hour_sin": np.sin(2 * np.pi * h / 24),
            "hour_cos": np.cos(2 * np.pi * h / 24)}


def _make_model():
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.Adam(LR))


def main():
    df = pd.read_csv((NB / cfg["path_data"]).resolve())
    df["ts"] = pd.to_datetime(df[cfg["time_col"]], unit="ms").dt.round("h")
    df = (df.groupby("ts", as_index=False)[cfg["target"]].mean()
            .dropna().sort_values("ts").reset_index(drop=True))
    seg = df.iloc[SEG_START:SEG_START + N_HOURS].reset_index(drop=True)
    ts = seg["ts"]

    # real slow trend = realistic wandering level
    level = (pd.Series(seg[cfg["target"]].to_numpy(float))
             .rolling(48, center=True, min_periods=1).mean().to_numpy())
    h = ts.dt.hour.to_numpy()
    amp = np.where(np.arange(N_HOURS) < SHIFT_AT, AMP_BEFORE, AMP_AFTER)
    rng = np.random.default_rng(0)
    y = level + amp * np.sin(2 * np.pi * (h - 15) / 24) + rng.normal(0, NOISE, N_HOURS)

    online, batch = _make_model(), _make_model()
    pred_o = np.full(N_HOURS, np.nan)
    pred_b = np.full(N_HOURS, np.nan)
    for t in range(N_HOURS):
        x = _features(ts[t])
        pred_o[t] = online.predict_one(x)
        pred_b[t] = batch.predict_one(x)
        online.learn_one(x, y[t])           # online: always learns
        if t < SHIFT_AT:                     # batch: frozen after the shift
            batch.learn_one(x, y[t])

    roll = lambda e: pd.Series(e).rolling(ROLL, min_periods=1).mean().to_numpy()
    roll_o = roll(np.abs(pred_o - y))
    roll_b = roll(np.abs(pred_b - y))

    x = np.arange(N_HOURS)
    ylim_top = (np.nanmin([y, pred_o, pred_b]) - 2, np.nanmax([y, pred_o, pred_b]) + 4)
    ylim_bot = (0, np.nanmax([roll_o, roll_b]) * 1.1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    def draw(t):
        axT.clear(); axB.clear()
        sl = slice(0, t + 1)
        axT.plot(x[sl], y[sl],      color="black",     lw=2.0, label="real")
        axT.plot(x[sl], pred_b[sl], color="steelblue", lw=1.4, ls="--", label="batch (frozen)")
        axT.plot(x[sl], pred_o[sl], color="tomato",    lw=1.4, ls="--", label="online (river)")
        axT.axvline(SHIFT_AT, color="0.5", ls=":", lw=1.5)
        axT.set_ylim(*ylim_top)
        axT.set_ylabel("temperature (degC)")
        axT.legend(loc="upper left", fontsize=9)
        axT.grid(alpha=0.25)
        if t >= SHIFT_AT:
            axT.text(SHIFT_AT, ylim_top[1], "  amplitude x%.1f" % (AMP_AFTER / AMP_BEFORE),
                     color="0.35", va="top", ha="left", fontsize=9)

        axB.plot(x[sl], roll_b[sl], color="steelblue", lw=1.8, label="batch error")
        axB.plot(x[sl], roll_o[sl], color="tomato",    lw=1.8, label="online error")
        axB.axvline(SHIFT_AT, color="0.5", ls=":", lw=1.5)
        axB.set_xlim(0, N_HOURS); axB.set_ylim(*ylim_bot)
        axB.set_ylabel(f"MAE {ROLL}h (degC)")
        axB.set_xlabel("hours")
        axB.legend(loc="upper left", fontsize=9)
        axB.grid(alpha=0.25)

        fig.suptitle("Daily cycle changes shape: online re-learns the new swing, "
                     "batch keeps the old shape", y=0.98)

    frames = list(range(50, N_HOURS, 5))
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS)
    anim.save(OUT_PATH, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"GIF written: {OUT_PATH}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()