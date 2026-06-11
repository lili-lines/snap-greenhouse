"""
Generate a GIF comparing ONE batch model vs ONE online (river) model,
fold by fold, for the project README.

Each frame = one test window (24 h): real temperature vs both models'
predictions, plus the per-fold MAE of each model. Playing through the folds
shows how the online model adapts over the seasons while the (frozen) batch
model drifts.

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
DPI         = 90             # lower = lighter GIF
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
    fig, ax = plt.subplots(figsize=(9, 4.5))

    def draw(i):
        ax.clear()
        s = splits[i]
        b, r = fb[s], fr[s]
        ax.plot(np.arange(len(b["y_test"])), b["y_test"],
                color="black", lw=2.2, label="real")
        ax.plot(np.arange(len(b["y_pred"])), b["y_pred"],
                color="steelblue", lw=1.6, ls="--", label=f"{BATCH_MODEL} (batch)")
        ax.plot(np.arange(len(r["y_pred"])), r["y_pred"],
                color="tomato", lw=1.6, ls="--", label=f"{RIVER_MODEL} (online)")

        mae_b = np.mean(np.abs(b["y_pred"] - b["y_test"]))
        mae_r = np.mean(np.abs(r["y_pred"] - r["y_test"]))
        # auto y-axis per fold: each 24 h window is framed on its own values
        vals = np.concatenate([b["y_test"], b["y_pred"], r["y_pred"]])
        pad = 0.08 * (vals.max() - vals.min() + 1e-9)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
        ax.set_xlabel("hour within the 24 h test window")
        ax.set_ylabel("temperature (degC)")
        ax.set_title(f"Fold {i + 1}/{len(splits)}  -  {b['label']}")
        ax.text(0.02, 0.96,
                f"MAE batch  = {mae_b:5.2f}\nMAE online = {mae_r:5.2f}",
                transform=ax.transAxes, va="top", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)

    anim = FuncAnimation(fig, draw, frames=len(splits), interval=1000 / FPS)
    anim.save(OUT_PATH, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"GIF written: {OUT_PATH}  ({len(splits)} folds)")


if __name__ == "__main__":
    main()
