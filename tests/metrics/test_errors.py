import numpy as np
import pandas as pd
from ocf_ml_metrics.metrics.errors import (
    compute_metrics,
    common_error_metrics,
    compute_error_part_of_year,
    compute_error_part_of_day,
)


def test_common_error_metrics():
    predictions = np.random.random((288, 1))
    target = np.random.random((288, 1))
    errors = common_error_metrics(predictions=predictions, target=target)
    for key in ["nmae", "mae", "rmse"]:
        assert key in errors.keys()
        assert isinstance(errors[key], float)


def test_compute_error_part_of_year():
    predictions = np.random.random((12, 1))
    target = np.random.random((12, 1))
    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-12-31 00:00", freq="1M")
    )
    errors = compute_error_part_of_year(predictions=predictions, target=target, datetimes=datetimes)
    for key in errors.keys():
        assert "Winter" in key or "Summer" in key or "Fall" in key or "Spring" in key


def test_compute_error_part_of_day():
    predictions = np.random.random((24, 1))
    target = np.random.random((24, 1))
    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-01-01 23:00", freq="1H")
    )
    errors = compute_error_part_of_day(predictions=predictions, target=target, datetimes=datetimes)
    for key in errors.keys():
        assert "Night" in key or "Morning" in key or "Afternoon" in key or "Evening" in key


def test_compute_metrics():
    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-12-31 23:00", freq="1H")
    )
    predictions = np.random.random((len(datetimes), 1))
    target = np.random.random((len(datetimes), 1))
    errors = compute_metrics(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        filter_by_night=True,
        latitude=55.3781,
        longitude=0.0,
        sun_position_for_night=-5,
    )
    assert len([key for key in errors if "no_night" in key]) == 27
    assert len([key for key in errors if "no_night" not in key]) == 27
    assert len([key for key in errors if "Winter" in key]) == 6  # night/no_night and 3 metrics each
    assert len([key for key in errors if "Summer" in key]) == 6
    assert len([key for key in errors if "Fall" in key]) == 6
    assert len([key for key in errors if "Spring" in key]) == 6
    assert len([key for key in errors if "Morning" in key]) == 6
    assert len([key for key in errors if "Afternoon" in key]) == 6
    assert len([key for key in errors if "Evening" in key]) == 6
    assert len([key for key in errors if "Night" in key]) == 6