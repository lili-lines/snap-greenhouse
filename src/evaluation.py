import pandas as pd
from sklearn.base import clone

from src.config import cfg
from src.cv import split_fold
from src.preprocess import prepare_raw, GreenhousePipeline
from src.journal import log_run
from src.metrics import compute_metrics
from src.river_model import build_model, train_online, predict_fold, _to_features, _add_lags

target   = cfg['target']
time_col = cfg['time_col']
lags     = cfg['lags']
windows  = cfg['rolling_windows']
patterns = cfg['feature_patterns']


def evaluate_folds(splits_info: pd.DataFrame,
                   model_name: str,
                   per_fold,
                   *,
                   params: dict,
                   features,
                   log: bool = True) -> pd.DataFrame:
    """
    - fold loop + scoring + log, shared by all evaluators (baseline, batch, river)
    - only behaviour that varies between models is producing the per-fold
    predictions: that's the responsibility of `per_fold`.

    Args:
        per_fold : callable(row) -> (y_true, y_pred, timestamp).
        features : list of features, or callable() -> list 
            (for river, we only know the features after processing the first fold)
    """
    rows = []          # scalars: ids + metrics + nb_data
    fold_result = []   # rows + raw arrays
    for _, row in splits_info.iterrows():
        y_true, y_pred, ts = per_fold(row)
        scalar = {
            "Split":  int(row["Split"]),
            "Month":  row["Month"],
            "Window": row["Window"],
            "model":  model_name,
            **compute_metrics(y_true, y_pred),
            "nb_data": len(y_true),
        }
        rows.append(scalar)
        fold_result.append({**scalar, "y_test": y_true, "y_pred": y_pred, "timestamp": ts})

    df_scores = pd.DataFrame(rows)
    if log and not df_scores.empty:
        mean = df_scores[["MAE", "RMSE", "R2", "MAPE"]].mean().round(4).to_dict()
        log_run(model_name=model_name,
                metrics_mean=mean,
                fold_result=fold_result,
                params=params,
                features=features() if callable(features) else features)
    return df_scores


def evaluate_baseline(splits_info: pd.DataFrame,
                      df_raw: pd.DataFrame,
                      horizon: int) -> pd.DataFrame:
    """Seasonal naive baseline: ŷ(t) = tempint(t - horizon hours)."""
    df = prepare_raw(df_raw, time_col)
    df["pred"] = df[target].shift(horizon)

    def per_fold(row):
        _, sub = split_fold(df, row, time_col)   # baseline don't use trainset
        sub = sub.dropna(subset=["pred"])
        return sub[target].values, sub["pred"].values, sub[time_col].values

    return evaluate_folds(
        splits_info, f"naive_{horizon}h", per_fold,
        params={"horizon": horizon}, features=[],
    )


def evaluate_batch(model,
                   model_name: str,
                   splits_info: pd.DataFrame,
                   df_raw: pd.DataFrame,
                   log: bool = True) -> pd.DataFrame:
    """
    evaluate a batch model (sklearn/tree) fold by fold
    + feature engineering via GreenhousePipeline
    """
    state = {"feat": []}

    def per_fold(row):
        df_raw_train, df_raw_test = split_fold(df_raw, row, time_col)
        pipeline = GreenhousePipeline(target_cols=[target],
                                      lag_hours=lags,
                                      rolling_windows=windows,
                                      datetime_col=time_col)
        df_train = pipeline.fit_transform(df_raw_train)
        df_test  = pipeline.transform(df_raw_test)

        feat = [c for c in df_train.columns if any(p in c for p in patterns)]
        state["feat"] = feat

        m = clone(model)
        m.fit(df_train[feat], df_train[target])
        y_pred = m.predict(df_test[feat])
        return df_test[target].values, y_pred, df_test[time_col].values

    return evaluate_folds(
        splits_info, model_name, per_fold,
        params=model.get_params(),
        features=lambda: state["feat"],
        log=log,
    )


def evaluate_river(splits_info: pd.DataFrame,
                   df_raw: pd.DataFrame,
                   model_kwargs: dict = None,
                   model_name: str = "river",
                   log: bool = True) -> pd.DataFrame:
    """
    evaluate River (online) model fold by fold
    """
    df_raw = prepare_raw(df_raw, time_col)

    def per_fold(row):
        df_train, df_test = split_fold(df_raw, row, time_col)
        model = build_model(**(model_kwargs or {}))
        train_online(model, df_train)
        y_pred = predict_fold(model, df_test, df_train)
        return df_test[target].values, y_pred, df_test[time_col].values

    return evaluate_folds(
        splits_info, model_name, per_fold,
        params=model_kwargs,
        features=lambda: list(_to_features(_add_lags(df_raw).iloc[-1]).keys()),
        log=log,
    )
