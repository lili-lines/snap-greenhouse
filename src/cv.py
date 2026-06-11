"""
cross-validation splits for time series data
with a bimonthly scheme (2 test windows per month)
"""
import pandas as pd


def bimonthly_splits(df: pd.DataFrame, 
                     datetime_col: str, 
                     start: str,
                     test_horizon: int,
                     save_path: str = None) -> pd.DataFrame:
    """ 2 windows of test with size = nb 'test_horizon' h (d8 & d22).
        train = all before
    """
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)

    months = pd.period_range(start=start,
                             end=df[datetime_col].max(),
                             freq="M"
                             )
    tz = df[datetime_col].dt.tz
    rows = []
    fold = 0
    for month in months:
        month_start = month.to_timestamp().tz_localize(tz) if tz else month.to_timestamp()
        anchors = [
            ("semaine-2", month_start + pd.Timedelta(days=7)),
            ("semaine-4", month_start + pd.Timedelta(days=21))
            ]
        for label, anchor in anchors:
            h_start = anchor
            h_end   = anchor + pd.Timedelta(hours=test_horizon)

            test_mask  = (df[datetime_col] >= h_start) & (df[datetime_col] < h_end)
            train_mask = df[datetime_col] < h_start

            if test_mask.sum() == 0:
                continue

            fold += 1
            col = f"fold_{fold}"
            df[col] = 0
            df.loc[test_mask, col] = 1

            rows.append({
                "Split":        fold,
                "Month":        str(month),
                "Window":       label,
                
                "Train Size":   int(train_mask.sum()),
                "Test Size":    int(test_mask.sum()),

                "train_start":  df.loc[train_mask, datetime_col].min(),
                "train_end":    df.loc[train_mask, datetime_col].max(),
                
                "test_start":   df.loc[test_mask, datetime_col].min(),
                "test_end":     df.loc[test_mask, datetime_col].max(),
            })

    if save_path:
        pd.DataFrame(rows).to_csv(save_path, index=False)
    return pd.DataFrame(rows)


def split_fold(df_raw, 
               row, 
               datetime_col, 
               unit="ms"):
    """
    return (df_train, df_test) for 1 fold defined by splits_info
    """
    ts      = pd.to_datetime(df_raw[datetime_col], unit=unit)
    t_start = pd.Timestamp(row["test_start"]).tz_localize(None)
    t_end   = pd.Timestamp(row["test_end"]).tz_localize(None)
    return (
        df_raw[ts < t_start],
        df_raw[(ts >= t_start) & (ts <= t_end)],
    )
