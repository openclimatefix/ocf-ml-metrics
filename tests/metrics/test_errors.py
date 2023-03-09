from typing import Final

import numpy as np
import pandas as pd

from ocf_ml_metrics.metrics.errors import (
    common_metrics,
    compute_metrics,
    compute_metrics_part_of_day,
    compute_metrics_part_of_year,
)

N_METRICS: Final[int] = 2  # mae and rmse


def test_common_error_metrics():
    predictions = np.random.random((288, 1))
    target = np.random.random((288, 1))
    errors = common_metrics(predictions=predictions, target=target)
    for key in ["mae", "rmse"]:
        assert key in errors.keys()
        assert isinstance(errors[key], float)


def test_compute_error_part_of_year():
    predictions = np.zeros((12, 1))

    # Set up different errors for different seasons, so we can check that the
    # code under test correctly slices up the seasons.
    target = np.zeros((12, 1))
    target[[11, 0, 1]] = 1  # Winter
    target[[2, 3, 4]] = 2  # Spring
    target[[5, 6, 7]] = 3  # Summer
    target[[8, 9, 10]] = 4  # Fall

    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-12-31 00:00", freq="1M")
    )
    assert len(datetimes) == 12
    errors = compute_metrics_part_of_year(
        predictions=predictions, target=target, datetimes=datetimes
    )
    for key in errors.keys():
        assert "Winter" in key or "Summer" in key or "Fall" in key or "Spring" in key

    expected_errors = {"Winter/mae": 1, "Spring/mae": 2, "Summer/mae": 3, "Fall/mae": 4}
    for key, err in expected_errors.items():
        np.testing.assert_almost_equal(errors[key], err, err_msg=f"{key} is incorrect!")


def test_compute_error_part_of_day():
    predictions = np.random.random((24, 1))
    target = np.random.random((24, 1))
    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-01-01 23:00", freq="1H")
    )
    errors = compute_metrics_part_of_day(
        predictions=predictions, target=target, datetimes=datetimes
    )
    for key in errors.keys():
        assert "Night" in key or "Morning" in key or "Afternoon" in key or "Evening" in key


def test_compute_metrics():
    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq="1H")
    )
    predictions = np.random.random((len(datetimes), 1))
    target = np.random.random((len(datetimes), 1))
    start_time = datetimes - pd.Timedelta("5min")
    errors = compute_metrics(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        filter_by_night=True,
        latitude=55.3781,
        longitude=0.0,
        sun_position_for_night=-5,
        start_time=start_time,
    )
    assert len([key for key in errors if "no_night" in key]) == 33 * N_METRICS
    assert len([key for key in errors if "no_night" not in key]) == 10 * N_METRICS
    assert (
        len([key for key in errors if "Winter" in key]) == 2 * N_METRICS
    )  # night/no_night and 2 metrics each
    assert len([key for key in errors if "Summer" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Fall" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Spring" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Morning" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Afternoon" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Evening" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Night" in key]) == 2 * N_METRICS


def test_compute_metrics_threshold():
    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq="1H")
    )
    predictions = np.random.random((len(datetimes), 1))
    target = np.random.random((len(datetimes), 1))
    start_time = datetimes - pd.Timedelta("5min")
    errors = compute_metrics(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        filter_by_night=True,
        latitude=55.3781,
        longitude=0.0,
        sun_position_for_night=-5,
        start_time=start_time,
        thresholds=[1000, 2000],  # MW
    )
    assert len([key for key in errors if "no_night" in key]) == 34 * N_METRICS
    assert len([key for key in errors if "no_night" not in key]) == 11 * N_METRICS
    assert (
        len([key for key in errors if "Winter" in key]) == 2 * N_METRICS
    )  # night/no_night and 2 metrics each
    assert len([key for key in errors if "Summer" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Fall" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Spring" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Morning" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Afternoon" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Evening" in key]) == 2 * N_METRICS
    assert len([key for key in errors if "Night" in key]) == 2 * N_METRICS
