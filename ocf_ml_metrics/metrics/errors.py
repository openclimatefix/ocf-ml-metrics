"""Common metrics to compute"""
from typing import Optional, Union

import numpy as np
import pandas as pd

from ocf_ml_metrics.metrics.utils import filter_night


def common_metrics(predictions: np.ndarray, target: np.ndarray, tag: str = "", **kwargs) -> dict:
    """
    Common error metrics.

    Computes RMSE and MAE. If you'd like to compute the normalized MAE (NMAE), then please
    normalize both the predictions and target before passing them into this function.

    Args:
        predictions: Predictions for the given time period.
        target: Ground truth to compare against, or output from baseline model.
        tag: Tag to add to the dictionary keys, if wanted.
        kwargs: Not used.

    Returns:
        Dictionary of error metrics computed over the given data.
    """
    error_dict = {}

    def _mean(input):
        # 2+ dimensional input - compute mean but preserve 0th dimension, yields 1-d ndarray
        if len(predictions.shape) > 1:
            return np.mean(input, axis=0)
        else:
            return np.mean(input)

    error_dict[tag + "mae"] = _mean(np.abs(predictions - target))
    error_dict[tag + "rmse"] = np.sqrt(_mean(np.square(predictions - target)))

    return error_dict


def compute_metrics_part_of_day(
    predictions: np.ndarray,
    target: np.ndarray,
    datetimes: np.ndarray,
    hour_split: Optional[dict[str, tuple[int, ...]]] = None,
    **kwargs,
) -> dict:
    """
    Compute error based on the time of day.

    Args:
        predictions: Prediction Array.
        target: Target array.
        datetimes: Array of datetimes.
        hour_split: Hour split. If None then a sensible default will be used.
        kwargs: Not used.

    Returns:
        Error dictionary based on the part of day.
    """
    if hour_split is None:
        hour_split = {
            "Night": (21, 22, 23, 0, 1, 2, 3),
            "Morning": (4, 5, 6, 7, 8, 9),
            "Afternoon": (10, 11, 12, 13, 14, 15),
            "Evening": (16, 17, 18, 19, 20),
        }

    metrics = {}
    for split, hours in hour_split.items():
        split_dates = np.asarray(
            [i for i, d in enumerate(datetimes) if pd.Timestamp(d).hour in hours]
        )
        if len(split_dates) > 0:
            metrics.update(
                common_metrics(predictions[split_dates], target[split_dates], tag=split + "/")
            )
    return metrics


def compute_metrics_part_of_year(
    predictions: np.ndarray,
    target: np.ndarray,
    datetimes: np.ndarray,
    year_split: Optional[dict[str, tuple[int, ...]]] = None,
    **kwargs,
) -> dict:
    """
    Compute error based on year split.

    Args:
        predictions: Prediction array.
        target: Target array.
        datetimes: Datetimes of targets/predictions.
        year_split: How to split the year. If None then a sensible default will be used.
        kwargs: Not used.

    Returns:
        Error based on the different times of year.
    """
    if year_split is None:
        year_split = {
            "Winter": (12, 1, 2),
            "Spring": (3, 4, 5),
            "Summer": (6, 7, 8),
            "Fall": (9, 10, 11),
        }

    metrics = {}
    for split, months in year_split.items():
        split_dates = np.asarray(
            [i for i, d in enumerate(datetimes) if pd.Timestamp(d).month in months]
        )
        if len(split_dates) > 0:
            metrics.update(
                common_metrics(predictions[split_dates], target[split_dates], tag=split + "/")
            )
    return metrics


def compute_metrics_time_horizons(
    predictions: np.ndarray,
    target: np.ndarray,
    datetimes: np.ndarray,
    start_time: np.ndarray,
    **kwargs,
) -> dict:
    """
    Compute error based on time horizons.

    Args:
        predictions: Prediction array.
        target: Target array.
        datetimes: Datetimes of targets/predictions.
        start_time: Datetimes of start time, to compute time deltas.
        kwargs: Not used.

    Returns:
        Error based on the different time horizons.
    """
    metrics = {}
    for i in range(len(datetimes)):
        time_delta: pd.Timedelta = pd.Timestamp(datetimes[i]) - pd.Timestamp(start_time[i])
        metrics.update(
            common_metrics(
                predictions[i],
                target[i],
                tag=f"forecast_horizon_{time_delta.seconds // 60}_minutes/",
            )
        )
    return metrics


def count_large_errors(
    predictions: np.ndarray,
    target: np.ndarray,
    threshold: float = -1.0,
    sigma: float = -1.0,
    **kwargs,
) -> dict:
    """
    Count large errors in forecast.

    Args:
        predictions: Prediction array.
        target: Target array.
        threshold: Threshold in absolute value, if >= 0,
            only one of threshold or sigma can be set.
        sigma: Sigma level for which counts as a large error, if >= 0
            only one of threshold or sigma can be set.
        **kwargs: Only the 'tag' key is used. 'tag' is optional.

    Returns:
        Error dictionary with the counts of the large errors.
    """
    if threshold >= 0:
        assert sigma < 0, ValueError("Cannot set both sigma and threshold")
    elif sigma >= 0:
        assert threshold < 0, ValueError("Cannot set both sigma and threshold")
        raise NotImplementedError("Sigma support isn't created yet")

    # Need error by forecast horizon
    errors = np.abs(predictions - target)
    large_error_count = 0
    for err in errors:
        if threshold >= 0:
            if err > threshold:
                large_error_count += 1

    error_dict = {
        kwargs.get("tag", "")
        + "large_error_count_"
        + f"{'threshold' if threshold >= 0 else 'sigma'}_"
        + f"{threshold if threshold >= 0 else sigma}": large_error_count
    }

    return error_dict


def compute_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    datetimes: np.ndarray,
    start_time: np.ndarray,
    filter_by_night: bool = False,
    tag: str = "",
    thresholds: Union[list, float] = -1,
    **kwargs,
) -> dict:
    """
    Convenience function to compute all metrics.

    Args:
        predictions: Prediction array.
        target: Target array.
        datetimes: Datetimes for the array.
        start_time: Array where predictions begin from,
            where the forecast time horizon is measured from.
        tag: Tag to use for overall (i.e. train/val/test).
        filter_by_night: Filter by night time as well and return metrics for only daytime,
            requires 'latitude'm 'longitude', and 'sun_position_for_night' kwargs.
        thresholds: Thresholds for computing large errors (i.e. what defines large)
            can be list of thresholds, or a single one. None are calculated by default.
        **kwargs: Kwargs for other options, like hour split, or year split.

    Returns:
        Dictionary of metrics
    """
    assert len(predictions) == len(target) == len(datetimes) == len(start_time), ValueError(
        "Lens of prediction, target, datetime, and start times "
    )
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
            predictions=predictions,
            target=target,
            datetimes=datetimes,
            start_time=start_time,
            **kwargs,
        )
    )
    if isinstance(thresholds, list):
        for thresh in thresholds:
            metrics.update(
                count_large_errors(predictions=predictions, target=target, threshold=thresh)
            )
    elif thresholds >= 0:
        metrics.update(
            count_large_errors(predictions=predictions, target=target, threshold=thresholds)
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
                predictions=day_predictions,
                target=day_target,
                datetimes=day_datetime,
                start_time=start_time,
                **kwargs,
            )
        )
        if isinstance(thresholds, list):
            for thresh in thresholds:
                day_metrics.update(
                    count_large_errors(
                        predictions=day_predictions, target=day_target, threshold=thresh
                    )
                )
        elif thresholds >= 0:
            day_metrics.update(
                count_large_errors(
                    predictions=day_predictions, target=day_target, threshold=thresholds
                )
            )
        for key in list(day_metrics.keys()):
            day_metrics["no_night/" + key] = day_metrics.pop(key)
        metrics.update(day_metrics)

    for key in list(metrics.keys()):
        metrics[tag + "/" + key] = metrics.pop(key)

    return metrics
