import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                         np.clip(np.abs(y_true) + np.abs(y_pred), 1e-8, None))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return mean_absolute_error(y_true, y_pred)

def mase(y_train, y_true, y_pred, m=12):
    y_train = np.asarray(y_train, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d_season = np.abs(y_train[m:] - y_train[:-m]).mean() if len(y_train) > m else np.nan
    denom = d_season if (d_season and not np.isnan(d_season) and d_season != 0) else 1e-8
    return np.abs(y_true - y_pred).mean() / denom
