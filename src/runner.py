import pandas as pd

from src.model_factory import build_models, river_configs
from src.evaluation import evaluate_baseline, evaluate_batch, evaluate_river


def run_all(cfg: dict,
            splits_info: pd.DataFrame,
            df_raw: pd.DataFrame,
            log: bool = True,
            verbose: bool = True) -> dict:
    """
    eval all yaml model (baseline + batch + online)
    return {name: df_scores}
    each run is logged if log=True
    """
    results = {}
    models_cfg = cfg['models']
    METRICS = ["MAE", "RMSE", "MAPE"]

    def _store(name, df_scores):
        """
        stores result & displays mean model metric
        """
        results[name] = df_scores
        if verbose:
            mean = df_scores[METRICS].mean()
            print("           " + "   ".join(f"{m}={mean[m]:7.3f}" for m in METRICS))

    # 1. batch (sklearn + tree)
    for name, model in build_models(models_cfg).items():
        if verbose:
            print(f"[batch]    {name}")
        _store(name, evaluate_batch(model, 
                                    name, 
                                    splits_info, 
                                    df_raw, 
                                    log=log))

    # 2. baseline.s
    for name, spec in models_cfg.items():
        if spec.get("type") != "baseline":
            continue
        if verbose:
            print(f"[baseline] {name}")
        _store(name, evaluate_baseline(splits_info, 
                                       df_raw, 
                                       horizon=spec["horizon"]))

    # 3. online (river)
    for name, kwargs in river_configs(models_cfg).items():
        if verbose:
            print(f"[online]   {name}  {kwargs}")
        _store(name, evaluate_river(splits_info, 
                                    df_raw, 
                                    model_kwargs=kwargs,
                                    model_name=name, 
                                    log=log))

    return results
