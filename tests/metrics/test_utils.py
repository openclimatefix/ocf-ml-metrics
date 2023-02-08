import numpy as np
import pandas as pd

from ocf_ml_metrics.metrics.utils import filter_night


def test_filter_night_sun_0_degrees():
    predictions = np.random.random((289, 1))  # 1 day at 5 minutely intervals
    target = np.random.random((289, 1))
    latitude = 55.3781  # Center of UK according to Google
    longitude = 3.4360
    datetimes = np.asarray(
        pd.date_range(start="2022-01-01 00:00", end="2022-01-02 00:00", freq="5min")
    )
    filtered_predictions, filtered_target, filtered_datetimes = filter_night(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        sun_position_for_night=0,
    )
    assert len(predictions) == len(target) == len(datetimes)
    assert len(filtered_target) == len(filtered_datetimes) == len(filtered_predictions)
    assert len(filtered_predictions) < len(predictions)
    assert len(filtered_target) < len(target)
    assert len(filtered_datetimes) < len(datetimes)


def test_filter_night_sun_multi_degrees():
    predictions = np.random.random((289, 1))  # 1 day at 5 minutely intervals
    target = np.random.random((289, 1))
    latitude = 55.3781  # Center of UK according to Google
    longitude = 3.4360
    datetimes = np.asarray(
        pd.date_range(start="2022-06-01 00:00", end="2022-06-02 00:00", freq="5min")
    )
    filtered_predictions, filtered_target, filtered_datetimes = filter_night(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        sun_position_for_night=0,
    )
    assert len(predictions) == len(target) == len(datetimes)
    assert len(filtered_target) == len(filtered_datetimes) == len(filtered_predictions)
    assert len(filtered_predictions) < len(predictions)
    assert len(filtered_target) < len(target)
    assert len(filtered_datetimes) < len(datetimes)
    filtered_predictions2, filtered_target2, filtered_datetimes2 = filter_night(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        sun_position_for_night=-5,
    )
    assert len(filtered_target2) == len(filtered_datetimes2) == len(filtered_predictions2)
    assert len(filtered_target) == len(filtered_datetimes) == len(filtered_predictions)
    assert len(filtered_predictions) < len(filtered_predictions2)
    assert len(filtered_target) < len(filtered_target2)
    assert len(filtered_datetimes) < len(filtered_datetimes2)
    filtered_predictions3, filtered_target3, filtered_datetimes3 = filter_night(
        predictions=predictions,
        target=target,
        datetimes=datetimes,
        latitude=latitude,
        longitude=longitude,
        sun_position_for_night=5,
    )
    assert len(filtered_target2) == len(filtered_datetimes2) == len(filtered_predictions2)
    assert len(filtered_target3) == len(filtered_datetimes3) == len(filtered_predictions3)
    assert len(filtered_predictions3) < len(filtered_predictions2)
    assert len(filtered_target3) < len(filtered_target2)
    assert len(filtered_datetimes3) < len(filtered_datetimes2)
