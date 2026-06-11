import numpy as np
import pandas as pd


def _time_features_from_ts(ts) -> dict:
    """
    calculate time feats
    """
    h   = ts.hour
    dow = ts.dayofweek
    doy = ts.dayofyear
    return {
        "hour":        h,
        "day_of_week": dow,
        "day_of_year": doy,
        "month":       ts.month,
        "quarter":     ts.quarter,
        "hour_sin":    np.sin(2 * np.pi * h   / 24),
        "hour_cos":    np.cos(2 * np.pi * h   / 24),
        "day_sin":     np.sin(2 * np.pi * dow / 7),
        "day_cos":     np.cos(2 * np.pi * dow / 7),
        "doy_sin":     np.sin(2 * np.pi * doy / 365),
        "doy_cos":     np.cos(2 * np.pi * doy / 365),
    }


def add_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    sklearn/preprocess : add time feats in df
    """
    feats = df[datetime_col].apply(_time_features_from_ts)
    return pd.concat([df, pd.DataFrame(feats.tolist(), index=df.index)], axis=1)


def time_features_dict(row: pd.Series, datetime_col: str) -> dict:
    """
    River : return dict of features from a row
    """
    return _time_features_from_ts(row[datetime_col])


def create_lag_features(df: pd.DataFrame, target_cols: list, lags: list) -> pd.DataFrame:
    """
    lag feats (t - lag) in df
    ok for batch (preprocess) & online (river)
    """
    for col in target_cols:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_cols: list, windows: list,
                         min_periods: int = 1) -> pd.DataFrame:
    """
    feats rolling mean & std in df
    ok for batch (preprocess) & online (river)
    """
    for col in target_cols:
        for w in windows:
            roll = df[col].rolling(window=w, min_periods=min_periods)
            df[f"{col}_rollma_{w}h"]  = roll.mean()
            df[f"{col}_rollstd_{w}h"] = roll.std()
    return df