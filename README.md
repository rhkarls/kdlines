# kdlines
[![pypi_shield](https://img.shields.io/pypi/v/kdlines.svg)](https://pypi.org/project/kdlines/)
[![pypi_license](https://badgen.net/pypi/license/kdlines/)](https://pypi.org/project/kdlines/)
![tests_workflow](https://github.com/rhkarls/kdlines/actions/workflows/run_flake8_pytest.yml/badge.svg)
Density heatmaps of many (time-)series using kernel density estimation using scikit-learn.

This package is at alpha stage and experimental.

## Requirements

    scikit-learn
    numpy
    matplotlib

## Installation

`pip install kdlines`
## Basic example

```python
"""
A simple example based on numpy arrays and pandas DataFrame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kdlines import KernelDensityLines

# generate test data, 1000 series each of length 100
x_n = 100
y_n = 1000

# resolution of the heatmap y axis (i.e. number of grid cells)
y_res = 100

# arrays and generated data
x = np.linspace(0, x_n, x_n)
ys = np.empty((y_n, x_n))
randg = np.random.default_rng(seed=645)
for i in range(y_n):
    ys[i] = np.sin(-x / 3) + 0.5 * randg.standard_normal(1) + x * 0.1 + 11

# Data frames with datetime index to simulate time series with timestamps
ys_df = pd.DataFrame(index=x, data=ys.T)
ys_df.index = pd.date_range("2020-01-01", periods=x_n, freq="D")

# Estimation on the pandas DataFrame
kde_df = KernelDensityLines(
    y_res=y_res, kde_kernel='linear', bandwidth='scott'
)
kde_df.fit(y_lines=ys_df)

# Plotting as subplot
# sharex is False in this case since imshow is not plotted, only labeled, 
# with timestamps
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharey=True, sharex=False)
axs[0].plot(ys_df.index, ys_df.to_numpy(), lw=1)
kde_df.plot(ax=axs[1])
fig.tight_layout()
```
![example_kde_df](https://github.com/rhkarls/kdlines/blob/main/examples/example_simple_kde_df.png)
