import numpy as np
from sklearn.metrics import mean_squared_log_error


def rmse(ytrue, ypred):
    return np.sqrt(mean_squared_log_error(ytrue, ypred))
