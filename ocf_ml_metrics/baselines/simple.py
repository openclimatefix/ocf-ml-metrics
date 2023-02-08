"""Simple baselines to use"""
import numpy as np


def zero_baseline(predictions: np.ndarray, **kwargs) -> np.ndarray:
    """
    Baseline of predicting all 0's

    Args:
        predictions: Prediction array, to match the baseline to

    Returns:
        Array of all 0's
    """
    return np.zeros_like(predictions)


def max_baseline(predictions: np.ndarray, max_capacity: float, **kwargs) -> np.ndarray:
    """
    Baseline of predicting the maximum capacity for all timesteps

    Args:
        predictions: Prediction array, to match the baseline to
        max_capacity: The max capacity of the system being predicted for

    Returns:
        Array of all max capacity
    """
    return np.ones_like(predictions) * max_capacity


def last_value_persistence(predictions: np.ndarray, last_value: float, **kwargs) -> np.ndarray:
    """
    Baseline of predicting the last value for all future timesteps

    Args:
        predictions: Prediction array, to match the baseline to
        last_value: The t0 value for the prediction

    Returns:
        Array of the last t0 value
    """
    return np.ones_like(predictions) * last_value


def last_day_persistence(predictions: np.ndarray):
    """
    Persistence by taking the vallues from the day before

    Args:
        predictions: Prediction array

    Returns:
        The last day persistence prediction
    """
    return NotImplementedError("Last Day persistence has not been added yet")
