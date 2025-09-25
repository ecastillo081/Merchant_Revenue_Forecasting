import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# define periods separating train and test data
H = 12

### create forecasting functions
def naive_forecast(y_train, h):
    return np.repeat(y_train.iloc[-1], h)

def seasonal_naive_forecast(y_train, h, m):
    return np.resize(y_train.iloc[-m:].values, h)

def sma_forecast(y_train, h, window=3):
    last = y_train.rolling(window).mean().iloc[-1]
    return np.repeat(last, h)

def wma_forecast(y_train, h, weights=(1, 2, 3)):
    w = np.array(weights)
    last = (y_train.iloc[-len(w):].values * w).sum() / w.sum()
    return np.repeat(last, h)

def linear_regression_forecast(df_merchant, h):
    """
    Uses marketing_spend, promo_month, macro_index as regressors.
    """
    g = df_merchant.copy().sort_values("date").set_index("date").asfreq("MS")
    y = g["revenue"].astype(float)

    # feature engineering
    g["rev_lag1"] = g["revenue"].shift(1)
    g["rev_lag12"] = g["revenue"].shift(12)
    g["month"] = g.index.month
    month_dummies = pd.get_dummies(g["month"], prefix="m", drop_first=True)
    X = pd.concat(
        [g[["marketing_spend","promo_month","macro_index","rev_lag1","rev_lag12"]], month_dummies],
        axis=1
    ).fillna(method="bfill").fillna(0)

    y_train, y_test = y.iloc[:-h], y.iloc[-h:]
    Xtr, Xte = X.iloc[:-h], X.iloc[-h:]

    lin = LinearRegression().fit(Xtr.values, y_train.values)
    yhat = lin.predict(Xte.values)
    return yhat
