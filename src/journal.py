import json
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.config import cfg
test_horizon = cfg['test_horizon']
target   = cfg['target']
path_xps = cfg['path_xps']


def log_run(
        model_name: str,
        metrics_mean: dict,
        fold_result: dict,
        params: dict,
        features: list,
        ) -> None:
    """
    add run in json journal
    """
    path = Path(path_xps)
    runs = json.loads(path.read_text()) if path.exists() else []

    def _serialize(v):
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        return str(v)
    
    # fold complet
    fold_name = f"../outputs/fold_result_{model_name}.json"
    pd.DataFrame(fold_result).to_json(fold_name,
                                      orient="records",
                                      indent=4)
    # resume
    t  = datetime.now().strftime("%Y-%m-%d %H:%M")
    runs.append({
        "date":         t,
        "model":        model_name,
        "test_horizon": test_horizon,
        "metrics_mean": metrics_mean,
        "fold_result":  fold_name,
        "params":       {k: _serialize(v) for k, v in (params or {}).items()},
        "target":       target,
        "n_features":   len(features),
        "features":     features,
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs, 
                               indent=2, 
                               ensure_ascii=False))

