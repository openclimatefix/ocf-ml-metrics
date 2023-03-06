# ocf-ml-metrics
Collection of simple baseline models and metrics for standardized evaluation of OCF forecasting models.

This package computes a variety of baselines and error metrics
including persistence of both last value and last day of generation,
comparison to PVLive, and max and zero baselines.

## Installation

Install with `pip install ocf-ml-metrics`

## Usage

The easiest way to use this package is to use the
convenience function `ocf_ml_metrics.metrics.errors.compute_metrics` that
computes all the basic error metrics overall, with and without night time,
for different parts of the year, different times of day, and all forecast
horizons by default.

There is also `ocf_ml_metrics.evaluation.evaluation.evaluation` that computes metrics after taking in a pandas
dataframe. These metrics are computed for the raw values, normalized values, against simple baseline models,
and per ID in the input dataframe. The input dataframe required data can be found in the docstring

And example usage would be

```python
from ocf_ml_metrics.evaluation.evaluation import evaluation
import pandas as pd

results_df = pd.read_csv('<path to csv>')

metrics: dict = evaluation(results_df=results_df, model_name='<model name>')
```
