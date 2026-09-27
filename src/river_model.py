import pandas as pd
import numpy as np
from river import linear_model, preprocessing, optim

from src.config import cfg
from src.features import time_features_dict, create_lag_features, add_rolling_features

target   = cfg['target']
time_col = cfg['time_col']
rolling_windows = cfg['rolling_windows']
lags = cfg['lags']


def _to_features(row: pd.Series) -> dict:
    """
    River feat dict for 1 row, driven by config 'lags' & 'rolling_windows'
    """
    feats = {
        **time_features_dict(row, "timestamp"),
        "is_weekend": int(row["timestamp"].dayofweek >= 5),
    }
    # 1 feature per config lag
    for lag in lags:
        feats[f"lag_{lag}h"] = row.get(f"{target}_lag_{lag}h", 0.0)
    # rolling mean + std per config window
    for w in rolling_windows:
        feats[f"rollma_{w}h"]  = row.get(f"{target}_rollma_{w}h",  0.0)
        feats[f"rollstd_{w}h"] = row.get(f"{target}_rollstd_{w}h", 0.0)
    # trend between consecutive lags (e.g. lag24 - lag48)
    for a, b in zip(sorted(lags), sorted(lags)[1:]):
        feats[f"diff_{a}_{b}h"] = (row.get(f"{target}_lag_{a}h", 0.0)
                                   - row.get(f"{target}_lag_{b}h", 0.0))
    return feats


def build_model(lr: float = 0.01, 
                l2: float = 0.0, 
                optimizer: str = "sgd"):
    """
    model River : StandardScaler + LinearRegression online
    - args:
        lr        : learning rate
        l2        : Ridge regul
        optimizer : 'sgd' | 'adam' | 'rmsprop'
    """
    _optimizers = {
        "sgd":     optim.SGD(lr),
        "adam":    optim.Adam(lr),
        "rmsprop": optim.RMSProp(lr),
    }
    opt = _optimizers.get(optimizer, optim.SGD(lr))
    return (
        preprocessing.StandardScaler()
        | linear_model.LinearRegression(optimizer=opt, l2=l2)
    )


def _add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = create_lag_features(df.copy(), 
                             [target], 
                             lags)
    df = add_rolling_features(df, 
                              [target], 
                              rolling_windows)
    return df.dropna(subset=[f"{target}_lag_{max(lags)}h"])


def train_online(model, df: pd.DataFrame, since=None):
    """
    train row by row on df
    🌊 `since`: only learn on rows strictly after this timestamp — lags/rolling
    are still computed on the full `df` (correct context), only the learning
    step is restricted, so incremental calls don't need a 48h context prefix.
    """
    rows = _add_lags(df)
    if since is not None:
        rows = rows[rows[time_col] > since]
    for _, row in rows.iterrows():
        model.learn_one(_to_features(row),
                        row[target])
    return model


def predict_and_learn_fold(model,
                           df_test: pd.DataFrame,
                           df_train: pd.DataFrame) -> np.ndarray:
    """
    pred on df_test with train context for lags
    🌊 true online loop: predict_one then learn_one on the ground truth,
    right after each prediction, so the model keeps adapting during the
    test window instead of staying frozen like a batch model.
    """
    context = pd.concat([df_train.tail(48),
                         df_test]).reset_index(drop=True)
    context  = _add_lags(context)
    df_test_with_lags = context.tail(len(df_test))

    preds = []
    for _, row in df_test_with_lags.iterrows():
        x = _to_features(row)
        preds.append(model.predict_one(x))
        model.learn_one(x, row[target])  # 🌊 learn immediately after predicting (no leakage: prediction already made)
    return np.array(preds)