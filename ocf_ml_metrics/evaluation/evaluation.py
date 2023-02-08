""" Evaluation the model results """

from typing import Union

import pandas as pd

from ocf_ml_metrics.baselines.simple import last_value_persistence, max_baseline, zero_baseline
from ocf_ml_metrics.evaluation.utils import check_results_df
from ocf_ml_metrics.metrics.errors import compute_metrics


def evaluation(
    results_df: pd.DataFrame,
    model_name: str,
    sun_threshold_degrees_for_night: float = -5.0,
    error_thresholds: Union[float, int, list] = [1000, 2000],
    outturn_unit: str = "mw",
    **kwargs,
) -> dict:
    """
    Main evaluation method

    1. checks results_df is in the correct format
    2. Makes evaluation of the metrics, both normalized and not
    3. Returns new dataframe with the results

    Args:
        results_df: results dataframe. This should have the following columns:
            - t0_datetime_utc
            - t0_actual_pv_outturn_[unit]
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
        kwargs: Arguments for compute_metrics to pass through
            (time_of_year dict, hour_split dict, etc.)

    """
    # make sure datetimes columns datetimes and floor target time t to nearest 5-minutes
    results_df["t0_datetime_utc"] = pd.to_datetime(results_df["t0_datetime_utc"])
    results_df["target_datetime_utc"] = pd.to_datetime(results_df["target_datetime_utc"])
    results_df["target_datetime_utc"] = results_df["target_datetime_utc"].dt.floor("5T")

    # check result format
    check_results_df(results_df, unit=outturn_unit)

    # Get component parts needed for compute metrics
    predictions = results_df[f"forecast_pv_outturn_{outturn_unit}"].to_numpy()
    target = results_df[f"actual_pv_outturn_{outturn_unit}"].to_numpy()
    datetimes = results_df["target_datetime_utc"].to_numpy()
    latitude = results_df["latitude"].to_numpy()
    longitude = results_df["longitude"].to_numpy()

    # Calculate metrics on raw outputs
    metrics = compute_metrics(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        sun_position_for_night=sun_threshold_degrees_for_night,
        thresholds=error_thresholds,
        tag=f"{model_name}" + f"/{outturn_unit}",
        filter_by_night=True,
        start_time=results_df["t0_datetime_utc"].to_numpy(),
        **kwargs,
    )

    # Calculate metrics on normalized outputs
    capacity = results_df[f"capacity_{outturn_unit}p"].to_numpy()
    metrics.update(
        compute_metrics(
            predictions=predictions / capacity,
            target=target / capacity,
            datetimes=datetimes,
            latitude=latitude,
            longitude=longitude,
            sun_position_for_night=sun_threshold_degrees_for_night,
            thresholds=error_thresholds,
            tag=f"{model_name}" + "/normalized",
            filter_by_night=True,
            start_time=results_df["t0_datetime_utc"].to_numpy(),
            **kwargs,
        )
    )

    # Calculate simple metrics baselines
    t0_outturn = results_df[f"t0_actual_pv_outturn_{outturn_unit}"].to_numpy()
    for baseline_name, baseline in [
        ("zero_baseline", zero_baseline),
        ("max_baseline", max_baseline),
        ("last_value_persistence_baseline", last_value_persistence),
    ]:
        metrics.update(
            compute_metrics(
                predictions=baseline(
                    predictions=predictions, max_capacity=capacity, last_value=t0_outturn
                ),
                target=target,
                datetimes=datetimes,
                latitude=latitude,
                longitude=longitude,
                sun_position_for_night=sun_threshold_degrees_for_night,
                thresholds=error_thresholds,
                tag=f"{baseline_name}" + f"/{outturn_unit}",
                filter_by_night=True,
                start_time=results_df["t0_datetime_utc"].to_numpy(),
                **kwargs,
            )
        )

        # Calculate metrics on normalized outputs
        capacity = results_df[f"capacity_{outturn_unit}p"].to_numpy()
        metrics.update(
            compute_metrics(
                predictions=baseline(
                    predictions=predictions / capacity,
                    max_capacity=capacity / capacity,
                    last_value=t0_outturn / capacity,
                ),
                target=target / capacity,
                datetimes=datetimes,
                latitude=latitude,
                longitude=longitude,
                sun_position_for_night=sun_threshold_degrees_for_night,
                thresholds=error_thresholds,
                tag=f"{baseline_name}" + "/normalized",
                filter_by_night=True,
                start_time=results_df["t0_datetime_utc"].to_numpy(),
                **kwargs,
            )
        )

    # Now do it per ID
    metrics.update(
        evaluation_per_id(
            results_df=results_df,
            model_name=model_name,
            sun_threshold_degrees_for_night=sun_threshold_degrees_for_night,
            error_thresholds=error_thresholds,
            outturn_unit=outturn_unit,
            **kwargs,
        )
    )

    # Metrics now contains all results, both normalized and raw
    # This should be returned as 2 pandas dataframe so it can be written as CSV somewhere
    # Output 1 is single forecast error metrics
    # Output 2 is summarized forecast errors (hour, time of year, etc.)
    # Output should be t0,forecast_time,forecast_error1-3,error vs baseline1-3

    return metrics


def evaluation_per_id(
    results_df: pd.DataFrame,
    model_name: str,
    sun_threshold_degrees_for_night: float = -5.0,
    error_thresholds: Union[float, int, list] = [1000, 2000],
    outturn_unit: str = "mw",
    **kwargs,
) -> dict:
    """
    Main evaluation method

    1. checks results_df is in the correct format
    2. Makes evaluation of the metrics, both normalized and not
    3. Returns new dataframe with the results

    Args:
        results_df: results dataframe. This should have the following columns:
            - t0_datetime_utc
            - t0_actual_pv_outturn_[unit]
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
        kwargs: Arguments for compute_metrics to pass through
            (time_of_year dict, hour_split dict, etc.)

    """
    # make sure datetimes columns datetimes and floor target time t to nearest 5-minutes
    results_df["t0_datetime_utc"] = pd.to_datetime(results_df["t0_datetime_utc"])
    results_df["target_datetime_utc"] = pd.to_datetime(results_df["target_datetime_utc"])
    results_df["target_datetime_utc"] = results_df["target_datetime_utc"].dt.floor("5T")

    # check result format
    check_results_df(results_df, unit=outturn_unit)
    metrics = {}

    # Go through it per ID
    for id in results_df["id"]:
        per_id_df = results_df.loc[results_df["id"] == id]
        # Get component parts needed for compute metrics
        predictions = per_id_df[f"forecast_pv_outturn_{outturn_unit}"].to_numpy()
        target = per_id_df[f"actual_pv_outturn_{outturn_unit}"].to_numpy()
        datetimes = per_id_df["target_datetime_utc"].to_numpy()
        latitude = per_id_df["latitude"].to_numpy()
        longitude = per_id_df["longitude"].to_numpy()

        # Calculate metrics on raw outputs
        metrics.update(
            compute_metrics(
                predictions=predictions,
                target=target,
                datetimes=datetimes,
                latitude=latitude,
                longitude=longitude,
                sun_position_for_night=sun_threshold_degrees_for_night,
                thresholds=error_thresholds,
                tag=f"{model_name}/" + f"id_{id}" + f"/{outturn_unit}",
                filter_by_night=True,
                start_time=per_id_df["t0_datetime_utc"].to_numpy(),
                **kwargs,
            )
        )

        # Calculate metrics on normalized outputs
        capacity = per_id_df[f"capacity_{outturn_unit}p"].to_numpy()
        metrics.update(
            compute_metrics(
                predictions=predictions / capacity,
                target=target / capacity,
                datetimes=datetimes,
                latitude=latitude,
                longitude=longitude,
                sun_position_for_night=sun_threshold_degrees_for_night,
                thresholds=error_thresholds,
                tag=f"{model_name}/" + f"id_{id}" + "/normalized",
                filter_by_night=True,
                start_time=per_id_df["t0_datetime_utc"].to_numpy(),
                **kwargs,
            )
        )

        # Calculate simple metrics baselines
        t0_outturn = per_id_df[f"t0_actual_pv_outturn_{outturn_unit}"].to_numpy()
        for baseline_name, baseline in [
            ("zero_baseline", zero_baseline),
            ("max_baseline", max_baseline),
            ("last_value_persistence_baseline", last_value_persistence),
        ]:
            metrics.update(
                compute_metrics(
                    predictions=baseline(
                        predictions=predictions, max_capacity=capacity, last_value=t0_outturn
                    ),
                    target=target,
                    datetimes=datetimes,
                    latitude=latitude,
                    longitude=longitude,
                    sun_position_for_night=sun_threshold_degrees_for_night,
                    thresholds=error_thresholds,
                    tag=f"{baseline_name}/" + f"id_{id}" + f"/{outturn_unit}",
                    filter_by_night=True,
                    start_time=per_id_df["t0_datetime_utc"].to_numpy(),
                    **kwargs,
                )
            )

            # Calculate metrics on normalized outputs
            capacity = per_id_df[f"capacity_{outturn_unit}p"].to_numpy()
            metrics.update(
                compute_metrics(
                    predictions=baseline(
                        predictions=predictions / capacity,
                        max_capacity=capacity / capacity,
                        last_value=t0_outturn / capacity,
                    ),
                    target=target / capacity,
                    datetimes=datetimes,
                    latitude=latitude,
                    longitude=longitude,
                    sun_position_for_night=sun_threshold_degrees_for_night,
                    thresholds=error_thresholds,
                    tag=f"{baseline_name}/" + f"id_{id}" + "/normalized",
                    filter_by_night=True,
                    start_time=per_id_df["t0_datetime_utc"].to_numpy(),
                    **kwargs,
                )
            )

    # Metrics now contains all results, both normalized and raw
    # This should be returned as 2 pandas dataframe so it can be written as CSV somewhere
    # Output 1 is single forecast error metrics
    # Output 2 is summarized forecast errors (hour, time of year, etc.)
    # Output should be t0,forecast_time,forecast_error1-3,error vs baseline1-3

    return metrics
