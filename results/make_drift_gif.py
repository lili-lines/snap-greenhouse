"""
GIF showing that ONLINE learning adapts to a SUDDEN regime change while a
FROZEN (batch) model stays wrong.

Controlled demo (concept drift), in 3 phases:
  1. TRAINING  — both models learn (green band);
  2. FROZEN, stable — batch stops learning, regime unchanged → still accurate (grey band);
  3. SHOCK — a sudden +OFFSET degC step: batch (frozen) can't follow, online re-learns.

- a continuous hourly slice of REAL greenhouse temperature (`tempint`);
- features = time-of-day only (hour sin/cos) → a level shift is invisible in the
  inputs, so it can only be followed by *learning*;
- two identical River models; batch is frozen after TRAIN_END, online keeps learning.

Output : results/figures/online_adapts_drift.gif
Run    : python results/make_drift_gif.py
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
SEG_START = 3000     # start of the continuous slice
N_HOURS   = 600      # length of the stream (25 days)
TRAIN_END = 240      # batch stops learning here (day 10)
SHIFT_AT  = 360      # sudden change happens here (day 15)
OFFSET    = 8.0      # magnitude of the sudden change (degC)
LR        = 0.1      # river Adam learning rate
ROLL      = 24       # rolling-error window (h)
OUT_PATH  = ROOT / "results" / "figures" / "online_adapts_drift.gif"
FPS       = 12
DPI       = 90

# modern palette (+ transparency used throughout)
C_REAL, C_BATCH, C_ONLINE = "#212529", "#3a86ff", "#ff006e"
C_TRAIN, C_FROZEN, C_SHOCK = "#52b788", "#adb5bd", "#d00000"
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

    y = seg[cfg["target"]].to_numpy(dtype=float).copy()
    y[SHIFT_AT:] += OFFSET   # <-- injected sudden regime change

    online, batch = _make_model(), _make_model()
    pred_o = np.full(N_HOURS, np.nan)
    pred_b = np.full(N_HOURS, np.nan)
    for t in range(N_HOURS):
        x = _features(ts[t])
        pred_o[t] = online.predict_one(x)
        pred_b[t] = batch.predict_one(x)
        online.learn_one(x, y[t])           # online: ALWAYS learns
        if t < TRAIN_END:                    # batch: frozen after TRAIN_END
            batch.learn_one(x, y[t])

    roll = lambda e: pd.Series(e).rolling(ROLL, min_periods=1).mean().to_numpy()
    roll_o = roll(np.abs(pred_o - y))
    roll_b = roll(np.abs(pred_b - y))

    xd = np.arange(N_HOURS) / 24.0          # x-axis in DAYS (clearer than raw hours)
    days = N_HOURS / 24.0
    d_train, d_shock = TRAIN_END / 24.0, SHIFT_AT / 24.0
    ylim_top = (np.nanmin([y, pred_o, pred_b]) - 2, np.nanmax([y, pred_o, pred_b]) + 5)
    ylim_bot = (0, np.nanmax([roll_o, roll_b]) * 1.12)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    def _bands(ax, alpha):
        ax.axvspan(0, d_train, color=C_TRAIN, alpha=alpha, lw=0)
        ax.axvspan(d_train, days, color=C_FROZEN, alpha=alpha * 1.4, lw=0)
        ax.axvline(d_shock, color=C_SHOCK, ls="--", lw=1.6, alpha=0.8)

    def draw(t):
        axT.clear(); axB.clear()
        for ax in (axT, axB):
            ax.set_facecolor("#fbfbfe")
            for s in ax.spines.values():
                s.set_edgecolor("#dfe3e8")
        sl = slice(0, t + 1)

        # ---- top panel : real vs batch vs online ----
        _bands(axT, 0.11)
        axT.plot(xd[sl], y[sl],      color=C_REAL,   lw=2.3, alpha=0.9,  label="real  (tempint, °C)")
        axT.plot(xd[sl], pred_b[sl], color=C_BATCH,  lw=1.8, ls="--", alpha=0.95, label="batch  (frozen)")
        axT.plot(xd[sl], pred_o[sl], color=C_ONLINE, lw=1.8, ls="--", alpha=0.95, label="online  (River)")
        axT.set_xlim(0, days); axT.set_ylim(*ylim_top)
        axT.set_ylabel("temperature (°C)")
        axT.grid(alpha=0.18)
        axT.legend(loc="lower left", fontsize=8, framealpha=0.85, ncol=3)

        # phase labels
        yt = ylim_top[1]
        axT.text(d_train / 2, yt, "batch: training", color=C_TRAIN, fontsize=8.5,
                 fontweight="bold", va="top", ha="center", alpha=0.95)
        axT.text((d_train + days) / 2, yt, "batch: frozen", color="#6c757d", fontsize=8.5,
                 fontweight="bold", va="top", ha="center", alpha=0.95)
        if t >= SHIFT_AT:
            axT.text(d_shock, ylim_top[0], f"  sudden +{OFFSET:g}°C", color=C_SHOCK,
                     fontsize=8.5, fontweight="bold", va="bottom", ha="left")
        # data / features info box
        axT.text(0.985, 0.95,
                 "Target: tempint (indoor °C)\nFeatures: hour-of-day (sin/cos)",
                 transform=axT.transAxes, va="top", ha="right", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ced4da", alpha=0.9))

        # ---- bottom panel : rolling error ----
        _bands(axB, 0.08)
        axB.plot(xd[sl], roll_b[sl], color=C_BATCH,  lw=2.0, label="batch error")
        axB.plot(xd[sl], roll_o[sl], color=C_ONLINE, lw=2.0, label="online error")
        axB.fill_between(xd[sl], roll_b[sl], 0, color=C_BATCH,  alpha=0.10)
        axB.fill_between(xd[sl], roll_o[sl], 0, color=C_ONLINE, alpha=0.13)
        axB.set_xlim(0, days); axB.set_ylim(*ylim_bot)
        axB.set_ylabel("MAE 24 h (°C)"); axB.set_xlabel("days")
        axB.grid(alpha=0.18)
        axB.legend(loc="upper left", fontsize=8, framealpha=0.85)

        fig.suptitle("Sudden regime change — batch is frozen after training, "
                     "online keeps learning at every step",
                     y=0.98, fontsize=12, fontweight="bold")

    frames = list(range(40, N_HOURS, 5))
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS)
    anim.save(OUT_PATH, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"GIF written: {OUT_PATH}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()