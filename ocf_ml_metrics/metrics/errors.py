import numpy as np

def common_error_metrics(predictions: np.ndarray, target: np.ndarray, tag: str = ""):
    """
    Common error metrics base

    Computes RMSE, NMAE, MAE for

    Args:
        predictions: Predictions for the given time period
        target: Ground truth to compare against, or output from baseline model
        tag: Tag to add to the dictionary keys, if wanted

    Returns:
        Dictionary of error metrics compute over the given data
    """
    error_dict = {}

    error_dict[tag+"nmae"] = np.mean(np.abs(predictions - target))
    error_dict[tag+"mae"] = np.mean(np.square(predictions - target))
    error_dict[tag+"rmse"] = np.sqrt(np.mean(np.square(predictions - target)))

    # Now per timestep

    return error_dict