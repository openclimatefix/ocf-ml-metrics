"""Utility functions"""
from typing import Tuple

import numpy as np
import pvlib


def filter_night(
    predictions: np.ndarray,
    target: np.ndarray,
    datetimes: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    sun_position_for_night: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter the predictions and targets by night time

    Night time is defined by the sun position below the horizon for the
    location given

    Args:
        predictions: Prediction array
        target: Target array
        datetimes: Datetimes for each target time
        latitude: Latitude of the site for predictions
        longitude: Longitude of site for predictions
        sun_position_for_night: Elevation at which it is considered 'night'

    Returns:
        The predictions, targets, and datetimes for all non-nighttime predictions

    """
    solpos = pvlib.solarposition.get_solarposition(
        time=datetimes,
        latitude=latitude,
        longitude=longitude,
        # Which `method` to use?
        # pyephem seemed to be a good mix between speed and ease but causes
        # segfaults!
        # nrel_numba doesn't work when using multiple worker processes.
        # nrel_c is probably fastest but requires C code to be manually compiled:
        # https://midcdmz.nrel.gov/spa/
    )
    elevation_mask = solpos["elevation"] <= sun_position_for_night
    predictions = predictions[~elevation_mask]
    target = target[~elevation_mask]
    datetimes = datetimes[~elevation_mask]
    return predictions, target, datetimes
