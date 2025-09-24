import numpy as np

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

# linear regression to be added later
