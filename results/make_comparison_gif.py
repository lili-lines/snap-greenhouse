"""
Generate a GIF comparing ONE batch model vs ONE online (river) model,
fold by fold, for the project README.

Each frame = one test window (<=24 h): real temperature vs both models'
predictions, plus the per-fold MAE of each model. Playing through the folds
shows how each model behaves across the seasons.

Output : results/figures/batch_vs_river.gif
Run    : python results/make_comparison_gif.py
Deps   : matplotlib (PillowWriter) — no extra package needed.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401  (kept for parity with the notebooks' env)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# make `src` importable (this script lives in results/)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import cfg
from src.results import load_journal, best_per_model

# config.yaml stores data/journal paths relative to notebooks/ — resolve them
# against that folder so this script runs from anywhere.
NB = ROOT / "notebooks"
def _resolve(rel):
    return (NB / rel).resolve()

# ============================== config ==============================
BATCH_MODEL = "Ridge_Poly2"   # editable: any batch model present in the journal
RIVER_MODEL = "river"         # editable: any online (river) run
OUT_PATH    = ROOT / "results" / "figures" / "batch_vs_river.gif"
FPS         = 1.5             # frames per second (lower = slower playback)
DPI         = 90              # lower = lighter GIF

# modern palette (shared with make_drift_gif.py for a coherent README look)
C_REAL, C_BATCH, C_ONLINE = "#212529", "#3a86ff", "#ff006e"
# ====================================================================


def load_folds(model_name, journal):
    """{Split: {'y_test', 'y_pred', 'label'}} for the best run of a model."""
    valid = journal[journal["fold_result"].apply(lambda p: _resolve(p).exists())]
    run = best_per_model(valid)
    matches = run.loc[run["model"] == model_name, "fold_result"]
    if matches.empty:
        raise SystemExit(f"Model '{model_name}' not found in the journal "
                         f"({cfg['path_xps']}). Run notebook 04 first.")
    folds = json.loads(_resolve(matches.iloc[0]).read_text(encoding="utf-8"))
    return {
        f["Split"]: {
            "y_test": np.asarray(f["y_test"], dtype=float),
            "y_pred": np.asarray(f["y_pred"], dtype=float),
            "label":  f"{f['Month']} - {f['Window']}",
        }
        for f in folds
    }


def main():
    journal = load_journal(str(_resolve(cfg["path_xps"])))
    fb = load_folds(BATCH_MODEL, journal)
    fr = load_folds(RIVER_MODEL, journal)

    splits = sorted(set(fb) & set(fr))   # folds available for BOTH models
    if not splits:
        raise SystemExit("No fold shared by the two models in the journal.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    def draw(i):
        ax.clear()
        ax.set_facecolor("#fbfbfe")
        for s in ax.spines.values():
            s.set_edgecolor("#dfe3e8")

        s = splits[i]
        b, r = fb[s], fr[s]
        xb, xr = np.arange(len(b["y_test"])), np.arange(len(r["y_pred"]))
        ax.plot(xb, b["y_test"], color=C_REAL,   lw=2.4, alpha=0.9,  label="real  (tempint, °C)")
        ax.plot(xb, b["y_pred"], color=C_BATCH,  lw=1.9, ls="--", alpha=0.95, label=f"{BATCH_MODEL}  (batch)")
        ax.plot(xr, r["y_pred"], color=C_ONLINE, lw=1.9, ls="--", alpha=0.95, label=f"{RIVER_MODEL}  (online)")
        # faint error fill of the online model vs its own real
        ax.fill_between(xr, r["y_pred"], r["y_test"], color=C_ONLINE, alpha=0.06)

        mae_b = np.mean(np.abs(b["y_pred"] - b["y_test"]))
        mae_r = np.mean(np.abs(r["y_pred"] - r["y_test"]))

        # auto y-axis per fold: each window is framed on its own values
        vals = np.concatenate([b["y_test"], b["y_pred"], r["y_pred"]])
        pad = 0.10 * (vals.max() - vals.min() + 1e-9)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
        ax.set_xlabel("hour within the test window")
        ax.set_ylabel("temperature (°C)")
        ax.set_title(f"Fold {i + 1}/{len(splits)}   ·   {b['label']}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.18)
        ax.legend(loc="lower center", fontsize=8.5, framealpha=0.85, ncol=3)

        # per-fold MAE scoreboard (winner in bold)
        win_b = mae_b <= mae_r
        ax.text(0.025, 0.95, f"MAE batch   {mae_b:4.2f} °C", transform=ax.transAxes,
                color=C_BATCH, va="top", family="monospace", fontsize=9.5,
                fontweight="bold" if win_b else "normal")
        ax.text(0.025, 0.86, f"MAE online  {mae_r:4.2f} °C", transform=ax.transAxes,
                color=C_ONLINE, va="top", family="monospace", fontsize=9.5,
                fontweight="bold" if not win_b else "normal")
        # data / models info box
        ax.text(0.985, 0.95,
                f"Target: tempint (indoor °C)\nbatch={BATCH_MODEL} · online={RIVER_MODEL}",
                transform=ax.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ced4da", alpha=0.9))

    anim = FuncAnimation(fig, draw, frames=len(splits), interval=1000 / FPS)
    anim.save(OUT_PATH, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"GIF written: {OUT_PATH}  ({len(splits)} folds)")


if __name__ == "__main__":
    main()