import numpy as np
from sklearn.metrics import mean_squared_error

def get_rmse(actual, prediction):
    return np.math.sqrt(mean_squared_error(actual, prediction))

def get_r_square(actual,prediction):
    return (1-get_rmse(actual,prediction))
