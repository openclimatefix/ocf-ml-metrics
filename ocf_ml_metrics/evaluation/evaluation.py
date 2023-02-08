""" Evaluation the model results """

import pandas as pd
from typing import Union
from ocf_ml_metrics.evaluation.utils import check_results_df
from ocf_ml_metrics.metrics.errors import compute_metrics


def evaluation(
    results_df: pd.DataFrame,
    model_name: str,
    sun_threshold_degrees_for_night: float = -5.0,
    error_thresholds: Union[float, int, list] = [1000, 2000],
    outturn_unit: str = "mw",
    **kwargs,
):
    """
    Main evaluation method

    1. checks results_df is in the correct format
    2. Makes evaluation of the metrics, both normalized and not
    3. Returns new dataframe with the results

    Args:
        results_df: results dataframe. This should have the following columns:
            - t0_datetime_utc
            - target_datetime_utc
            - forecast_pv_outturn_[unit]
            - actual_pv_outturn_[unit]
            - latitude
            - longitude
            - id
            - capacity_[unit]p
        where [unit] is set by outturn_unit, and is generally 'mw', 'kw', or 'w'
        model_name: the model name, used for adding titles to plots
        sun_threshold_degrees_for_night: Sun elevation degrees which signifies 'night'
        error_thresholds: Thresholds, in MW for 'large errors'
        outturn_unit: Str for the units used in the evaluation, usually 'mw', 'kw', or 'w'

    """
    # make sure datetimes columns datetimes and floor target time t to nearest 5-minutes
    results_df["t0_datetime_utc"] = pd.to_datetime(results_df["t0_datetime_utc"])
    results_df["target_datetime_utc"] = pd.to_datetime(results_df["target_datetime_utc"])
    results_df["target_datetime_utc"] = results_df["target_datetime_utc"].dt.floor("5T")

    # check result format
    check_results_df(results_df, unit=outturn_unit)

    # Get component parts needed for compute metrics
    predictions = results_df[f"forecast_pv_outturn_{outturn_unit}"]
    target = results_df[f"actual_pv_outturn_{outturn_unit}"]
    datetimes = results_df["target_datetime_utc"]
    latitude = results_df["latitude"]
    longitude = results_df["longitude"]

    # Calculate metrics on raw outputs
    metrics = compute_metrics(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        sun_position_for_night=sun_threshold_degrees_for_night,
        thresholds=error_thresholds,
        tag= f"{model_name}/" + results_df["id"] + f"/{outturn_unit}",
        filter_by_night=True,
        start_time=pd.Timestamp(results_df["t0_datetime_utc"]),
    )

    # Calculate metrics on normalized outputs
    capacity = results_df[f"capacity_{outturn_unit}p"]
    metrics.update(
        compute_metrics(
            predictions=predictions / capacity,
            target=target / capacity,
            datetimes=datetimes,
            latitude=latitude,
            longitude=longitude,
            sun_position_for_night=sun_threshold_degrees_for_night,
            thresholds=error_thresholds / capacity,
            tag=f"{model_name}/" +results_df["id"] + "/normalized",
            filter_by_night=True,
            start_time=pd.Timestamp(results_df["t0_datetime_utc"]),
        )
    )

    # Metrics now contains all results, both normalized and raw

