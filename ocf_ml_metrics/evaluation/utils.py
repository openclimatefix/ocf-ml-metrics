""" Util evaluation functions """
import pandas as pd


def check_results_df(results_df: pd.DataFrame, unit: str = "mw"):
    """
    Check the dataframe has the correct columns

    Args:
        results_df: results dataframe
        unit: Unit to check for, usually 'mw', 'kw', or 'w'

    """

    assert len(results_df) > 0
    assert "t0_datetime_utc" in results_df.keys()
    assert "target_datetime_utc" in results_df.keys()
    assert f"forecast_pv_outturn_{unit}" in results_df.keys()
    assert f"actual_pv_outturn_{unit}" in results_df.keys()
    assert f"t0_actual_pv_outturn_{unit}" in results_df.keys()
    assert "id" in results_df.keys()
    assert "latitude" in results_df.keys()
    assert "longitude" in results_df.keys()
    assert f"capacity_{unit}p" in results_df.keys()
