import numpy as np
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error,
                             r2_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    compute regression metrics: MAE, RMSE, R2, MAPE
    """
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }
