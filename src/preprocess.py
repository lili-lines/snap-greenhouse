import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import add_time_features, create_lag_features, add_rolling_features


def prepare_raw(df: pd.DataFrame, 
                datetime_col: str, 
                unit: str = "ms") -> pd.DataFrame:
    """ 
    convert ms → datetime, 
    + round h & mean/h & sort
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], unit=unit)

    df[datetime_col] = df[datetime_col].dt.round("h")
    df = df.groupby(datetime_col, as_index=False).mean(numeric_only=True)
    
    return df.sort_values(datetime_col).reset_index(drop=True)


def build_features(df: pd.DataFrame, 
                   target_cols: list = None,
                   datetime_col: str = None,
                   lag_hours: list = None, 
                   rolling_windows: list = None) -> pd.DataFrame:
    """
    features engineering : time → lags → rolling
    data prepare (1 row/h)
    """
    df = add_time_features(df, datetime_col=datetime_col)
    df = create_lag_features(df, target_cols=target_cols, lags=lag_hours)
    df = add_rolling_features(df, target_cols=target_cols, windows=rolling_windows)
    return df


class GreenhousePipeline:
    """
    feature engineering
    pipeline.fit(df_train)     → stores history for lags
    pipeline.transform(df)     → apply feats to df
    pipeline.fit_transform(df) → fit + transform
    """

    def __init__(self, target_cols: list,
                 lag_hours: list, 
                 rolling_windows: list, 
                 datetime_col: str):
        self.target_cols     = target_cols
        self.lag_hours       = lag_hours
        self.rolling_windows = rolling_windows
        self.datetime_col    = datetime_col
        self._lookback       = max(lag_hours + rolling_windows)
        self._history: pd.DataFrame = None

    def _build(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df, 
                              self.target_cols,
                              self.datetime_col,
                              self.lag_hours, 
                              self.rolling_windows
                              )

    def fit(self, df: pd.DataFrame):
        """ Keep history for lags & rolling windows"""
        df = prepare_raw(df, self.datetime_col)
        self._history = df.tail(self._lookback).copy()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = prepare_raw(df, self.datetime_col)
        if self._history is not None:
            df_ext = pd.concat([self._history, df], ignore_index=True)
            return self._build(df_ext).tail(len(df))
        return self._build(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)