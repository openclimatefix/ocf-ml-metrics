# ocf-ml-metrics
Collection of simple baseline models and metrics for standardized evaluation of OCF forecasting models.

This package computes a variety of baselines and error metrics
including persistence of both last value and last day of generation,
comparison to PVLive, and max and zero baselines. 


## Usage

The easiest way to use this package is to use the 
convenience function `ocf_ml_metrics.metrics.errors.compute_metrics` that
computes all the basic error metrics overall, with and without night time,
for different parts of the year, different times of day, and all forecast
horizons by default. 
