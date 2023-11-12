import numpy as np

def calculate_bias(y_true, y_pred):
    """Calculate bias (squared) for predicted values.

    Args:
        y_true: numpy array, true target values
        y_pred: numpy array, predicted values

    Returns:
        bias: scalar, bias squared
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")
    
    bias = np.mean((y_true - y_pred) ** 2)
    return bias

def calculate_variance(y_pred_list):
    """Calculate variance of predicted values.

    Args:
        y_pred_list: list of numpy arrays, predicted values for different data points

    Returns:
        variance: scalar, variance of predicted values
    """
    if not y_pred_list:
        raise ValueError("Input list must not be empty.")
    
    num_predictions = len(y_pred_list)
    prediction_dim = len(y_pred_list[0])
    
    # Stack predicted values for all data points into an array
    stacked_predictions = np.stack(y_pred_list)
    
    # Calculate the mean prediction for each data point
    mean_predictions = np.mean(stacked_predictions, axis=0)
    
    # Calculate the variance
    variance = np.mean((stacked_predictions - mean_predictions) ** 2)
    
    return variance
