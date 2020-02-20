import numpy as np
from sklearn.metrics import mean_squared_error

def get_RMSE(y, prediction):
    return np.math.sqrt(mean_squared_error(y, prediction))