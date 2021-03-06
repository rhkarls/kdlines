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

# Estimation on the numpy arrays
kde_np = KernelDensityLines(
    y_res=y_res, kde_kernel='linear', bandwidth='scott'
)
kde_np.fit(y_lines=ys, x=x)
# plot result
# note: depending in your settings, figure may not be shown automatically
# if not, call ax.get_figure().show()
ax = kde_np.plot()

# Estimation on the pandas DataFrame
kde_df = KernelDensityLines(
    y_res=y_res, kde_kernel='gaussian', bandwidth='scott'
)
kde_df.fit(y_lines=ys_df)

# Plotting as subplot
# sharex is False in this case since imshow is not plotted, only labeled,
# with timestamps
fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharey=True, sharex=False)
axs[0].plot(ys_df.index, ys_df.to_numpy(), lw=1)
kde_df.plot(ax=axs[1], cmap='inferno')
fig.tight_layout()
fig.show()

# Plotting with lines
fig2, ax2 = plt.subplots()
kde_df.plot(ax=ax2, cmap='summer', show_data=True)
fig2.show()

# Extracting density values from a "row" of the heatmap
kd_at_y = kde_df.kd_at_y(y=16)
fig3, ax3 = plt.subplots()
ax3.plot(ys_df.index, kd_at_y)
