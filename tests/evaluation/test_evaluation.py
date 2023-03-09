import pandas as pd

from ocf_ml_metrics.evaluation.evaluation import evaluation


def test_evaluation():
    t0 = pd.date_range(start="2022-01-01 00:00", end="2022-01-01 4:00", freq="1H")
    target_dt = pd.date_range(start="2022-01-01 5:00", end="2022-01-01 9:00", freq="1H")
    ids = [1, 1, 1, 2, 2]
    lat = [55.3781, 55.3781, 55.3781, 20.0, 20.0]
    longs = [0.0, 0.0, 0.0, 60.0, 60.0]
    forecast = [10000, 9000, 8500, 100, 150]
    actual = [8000, 8950, 8000, 101, 145]
    last_val = [9000, 3000, 5000, 100, 100]
    cap = [12000, 12000, 12000, 200, 2000]
    results_df = pd.DataFrame(
        {
            "t0_datetime_utc": t0,
            "target_datetime_utc": target_dt,
            "id": ids,
            "latitude": lat,
            "longitude": longs,
            "forecast_pv_outturn_mw": forecast,
            "actual_pv_outturn_mw": actual,
            "t0_actual_pv_outturn_mw": last_val,
            "capacity_mwp": cap,
        }
    )
    metrics = evaluation(results_df=results_df, model_name="test")
    assert len(metrics.keys()) == 432  # Lots of metrics
