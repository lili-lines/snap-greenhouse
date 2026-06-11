import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import compute_metrics


def load_journal(path: str) -> pd.DataFrame:
    """
    read xps_results.json → flattened df (1 row per run)
    - cols: date, model, metrics_mean.MAE, ... , fold_result
    """
    runs = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.json_normalize(runs)


def best_per_model(journal: pd.DataFrame, by: str = "metrics_mean.MAE") -> pd.DataFrame:
    """
    keep the best run (min metric) per model
    """
    return (journal.sort_values(by)
                   .groupby("model", as_index=False)
                   .first())


def global_metrics(fold_path: str) -> dict:
    """
    recompute MAE/RMSE/R²/MAPE on the CONCAT pred of all folds
    ⚠️ fixes the per-fold-R²-averaging trap: 
        an R² must be computed over the whole
        set of points, not averaged fold by fold
    """
    folds = json.loads(Path(fold_path).read_text(encoding="utf-8"))

    # fallback if predictions are not stored (old format):
    # mean of per-fold metrics (approximate R², to be avoided)
    if not folds or "y_test" not in folds[0]:
        df = pd.DataFrame(folds)
        return {
            "MAE":  df["MAE"].mean(),
            "RMSE": df["RMSE"].mean(),
            "R2":   df["R2"].mean(),
            "MAPE": df["MAPE"].mean(),
            "n":    int(df.get("nb_data", pd.Series([0])).sum()),
            "global": False,   # per-fold-averaged metrics, not global
        }
    y_true = np.concatenate([np.asarray(f["y_test"]) for f in folds])
    y_pred = np.concatenate([np.asarray(f["y_pred"]) for f in folds])
    return {
        **compute_metrics(y_true, y_pred),
        "n":    len(y_true),
        "global": True,    # true global metrics (concatenated predictions)
    }


def comparison_table(journal_path: str, 
                     baseline_model: str = "naive_24h") -> pd.DataFrame:
    """
    comparison table: best run per model, GLOBAL metrics (recomputed on
    concat pred) + relative gap vs baseline
    """
    journal = load_journal(journal_path)
    # keep only runs whose fold_result actually exists
    journal = journal[journal["fold_result"].apply(lambda p: Path(p).exists())]
    if journal.empty:
        raise FileNotFoundError(
            "Re-run notebooks 04 (batch/online/baseline) "
        )

    best = best_per_model(journal)
    rows = []
    for _, run in best.iterrows():
        g = global_metrics(run["fold_result"])
        rows.append({"model": run["model"], "date": run["date"], **g})
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    # relative MAE gap vs baseline
    if baseline_model in table["model"].values:
        base_mae = table.loc[table["model"] == baseline_model, "MAE"].iloc[0]
        table["vs_baseline_%"] = ((table["MAE"] - base_mae) / base_mae * 100).round(1)

    return table