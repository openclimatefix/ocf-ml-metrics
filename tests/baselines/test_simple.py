import numpy as np

from ocf_ml_metrics.baselines.simple import last_value_persistence, max_baseline, zero_baseline


def test_zero_baseline():
    predictions = np.random.random((100, 1))  # 100 random values
    baseline = zero_baseline(predictions=predictions)
    assert np.sum(baseline) == 0
    assert predictions.shape == baseline.shape


def test_max_baseline_max_of_predictions():
    predictions = np.random.random((100, 1))
    baseline = max_baseline(predictions=predictions, max_capacity=np.max(predictions))
    assert np.isclose(np.sum(baseline), (np.max(predictions) * len(predictions)))
    assert baseline.shape == predictions.shape


def test_max_baseline_max_manual_set():
    predictions = np.random.random((100, 1))
    baseline = max_baseline(predictions=predictions, max_capacity=1.0)
    assert np.isclose(np.sum(baseline), (1.0 * len(predictions)))
    assert baseline.shape == predictions.shape


def test_last_value_persistence():
    predictions = np.random.random((100, 1))
    t0_value = np.random.random()
    baseline = last_value_persistence(predictions=predictions, last_value=t0_value)
    assert np.isclose(np.sum(baseline), (t0_value * len(predictions)))
    assert baseline.shape == predictions.shape
