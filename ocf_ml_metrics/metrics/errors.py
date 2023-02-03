import numpy as np
import pandas as pd
from ocf_ml_metrics.utils import filter_night


def common_metrics(
        predictions: np.ndarray, target: np.ndarray, tag: str = "", **kwargs
) -> dict:
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

    error_dict[tag + "nmae"] = np.mean(np.abs(predictions - target))
    error_dict[tag + "mae"] = np.mean(np.square(predictions - target))
    error_dict[tag + "rmse"] = np.sqrt(np.mean(np.square(predictions - target)))

    # Now per timestep

    return error_dict


def compute_metrics_part_of_day(
        predictions: np.ndarray,
        target: np.ndarray,
        datetimes: np.ndarray,
        hour_split: dict = {
            "Night": (21, 22, 23, 0, 1, 2, 3),
            "Morning": (4, 5, 6, 7, 8, 9),
            "Afternoon": (10, 11, 12, 13, 14, 15),
            "Evening": (16, 17, 18, 19, 20),
        },
        **kwargs
) -> dict:
    """
    Compute error based on the time of day

    Args:
        predictions: Prediction Array
        target: Target array
        datetimes: Array of datetimes
        hour_split: Hour split

    Returns:
        Error dictionary based on the part of day
    """
    metrics = {}
    for split, hours in hour_split.items():
        split_dates = np.asarray(
            [i for i, d in enumerate(datetimes) if pd.Timestamp(d).hour in hours]
        )
        metrics.update(
            common_metrics(predictions[split_dates], target[split_dates], tag=split + "/")
        )
    return metrics


def compute_metrics_part_of_year(
        predictions: np.ndarray,
        target: np.ndarray,
        datetimes: np.ndarray,
        year_split: dict = {
            "Winter": (11, 0, 1),
            "Spring": (2, 3, 4),
            "Summer": (5, 6, 7),
            "Fall": (8, 9, 10),
        },
        **kwargs
) -> dict:
    """
    Compute error based on year split

    Args:
        predictions: Prediction array
        target: Target array
        datetimes: Datetimes of targets/predictions
        year_split: How to split the year

    Returns:
        Error based on the different times of year
    """
    metrics = {}
    for split, months in year_split.items():
        split_dates = np.asarray(
            [i for i, d in enumerate(datetimes) if pd.Timestamp(d).month in months]
        )
        metrics.update(
            common_metrics(predictions[split_dates], target[split_dates], tag=split + "/")
        )
    return metrics


def compute_metrics_time_horizons(
        predictions: np.ndarray,
        target: np.ndarray,
        datetimes: np.ndarray,
        start_time: pd.Timestamp,
        **kwargs
) -> dict:
    """
    Compute error based on time horizons

    Args:
        predictions: Prediction array
        target: Target array
        datetimes: Datetimes of targets/predictions
        start_time: Datetime of start time, to compute time deltas

    Returns:
        Error based on the different time horizons
    """
    metrics = {}
    for i in range(len(datetimes)):
        time_delta: pd.Timedelta = pd.Timestamp(datetimes[i]) - start_time
        metrics.update(
            common_metrics(predictions[i], target[i], tag=f"forecast_horizon_{time_delta.min}_minutes/")
        )
    return metrics


def compute_large_metrics(
        predictions: np.ndarray, target: np.ndarray, threshold: float, sigma: float, **kwargs
) -> dict:
    pass


def compute_metrics(
        predictions: np.ndarray,
        target: np.ndarray,
        datetimes: np.ndarray,
        filter_by_night: bool = False,
        tag: str = "",
        **kwargs
) -> dict:
    """
    Convience function to compute all metrics

    Args:
        predictions: Prediction array
        target: Target array
        datetimes: Datetimes for the array
        tag: Tag to use for overall (i.e. train/val/test)
        filter_by_night: Filter by night time as well and return metrics for only daytime,
            requires 'latitude'm 'longitude', and 'sun_position_for_night' kwargs
        **kwargs: Kwargs for other options, like hour split, or year split

    Returns:
        Dictionary of metrics
    """
    metrics = common_metrics(predictions=predictions, target=target)
    metrics.update(
        compute_metrics_part_of_day(
            predictions=predictions, target=target, datetimes=datetimes, **kwargs
        )
    )
    metrics.update(
        compute_metrics_part_of_year(
            predictions=predictions, target=target, datetimes=datetimes, **kwargs
        )
    )
    metrics.update(
        compute_metrics_time_horizons(
            predictions=predictions, target=target, **kwargs
        )
    )

    # Filter by night and run again
    if filter_by_night:
        day_predictions, day_target, day_datetime = filter_night(
            predictions=predictions,
            target=target,
            datetimes=datetimes,
            latitude=kwargs.get("latitude"),
            longitude=kwargs.get("longitude"),
            sun_position_for_night=kwargs.get("sun_position_for_night", -5),
        )
        day_metrics = common_metrics(predictions=day_predictions, target=day_target)
        day_metrics.update(
            compute_metrics_part_of_day(
                predictions=day_predictions, target=day_target, datetimes=day_datetime, **kwargs
            )
        )
        day_metrics.update(
            compute_metrics_part_of_year(
                predictions=day_predictions, target=day_target, datetimes=day_datetime, **kwargs
            )
        )
        day_metrics.update(
            compute_metrics_time_horizons(
                predictions=day_predictions, target=day_target, **kwargs
            )
        )
        for key in list(day_metrics.keys()):
            day_metrics["no_night/" + key] = day_metrics.pop(key)
        metrics.update(day_metrics)

    for key in list(metrics.keys()):
        metrics[tag + "/" + key] = metrics.pop(key)

    return metrics
