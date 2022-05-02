# -*- coding: utf-8 -*-
"""
pytest functions for kdlines

Expected results are stored in numpy npz file
https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez

np.savez('expected_grids.npz',
         grid_expected_1d=grid_expected_1d,
         grid_expected_2d=grid_expected_2d,
         kd_at_y_1d_expected=kd_at_y_1d_expected)

expected_grids = np.load('expected_grids.npz')
grid_expected_1d = expected_grids['grid_expected_1d']
grid_expected_2d = expected_grids['grid_expected_2d']
kd_at_y_1d_expected = expected_grids['kd_at_y_1d_expected']
    
"""

import string

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from kdlines import KernelDensityLines

# generate test data, 1000 series each of length 24
x_n = 24
y_n = 1000

y_res = 30
x = np.linspace(0, 24, x_n)
ys = np.empty((y_n, x_n))
randg = np.random.default_rng(seed=645)
for i in range(y_n):
    ys[i] = np.sin(-x / 3) + 0.1 * randg.standard_normal(1) + x * 0.1 + 11

# Data frames with numeric index, datetime index and string index
ys_df = pd.DataFrame(index=x, data=ys.T)

ys_df_di = ys_df.copy()
ys_df_di.index = pd.date_range("2020-01-01", "2020-01-24", freq="D")

ys_df_stri = ys_df.copy()
a_str = string.ascii_lowercase
a_list = list(a_str)[:24]
ys_df_stri.index = a_list

# some random 2d data
values_2d = np.vstack([randg.standard_normal(y_n), randg.standard_normal(y_n)])

# some fixed testing argument values
bandwidth = 0.25
kde_kernel = "linear"

# expected grids
expected_grids = np.load("expected_grids.npz")
grid_expected_1d = expected_grids["grid_expected_1d"]
grid_expected_2d = expected_grids["grid_expected_2d"]
kd_at_y_1d_expected = expected_grids["kd_at_y_1d_expected"]


def test_1d_kde():
    kde_1d = KernelDensityLines(
        y_res=y_res, kde_kernel=kde_kernel, bandwidth=bandwidth
    )
    kde_1d.fit(y_lines=ys, x=x)

    grid_result = kde_1d.kds
    grid_expected = grid_expected_1d

    assert np.allclose(grid_result, grid_expected)


def test_1d_vs_2d_gaussian_kde():
    # should produce very similar result on test data
    # also tests that their shapes are the same
    kde_1d = KernelDensityLines(
        y_res=y_res, kde_kernel="gaussian", bandwidth=bandwidth
    )

    kde_1d.fit(y_lines=ys, x=x)

    kde_2d = KernelDensityLines(
        y_res=y_res, kde_kernel="gaussian", bandwidth=bandwidth, dimensions=2
    )
    kde_2d.fit(y_lines=ys, x=x)

    assert np.allclose(kde_1d.kds, kde_2d.kds, atol=0.001)


def test_1d_kde_df():
    kde_1d = KernelDensityLines(
        y_res=y_res, kde_kernel="linear", bandwidth=bandwidth
    )
    kde_1d.fit(y_lines=ys_df, x=None)

    grid_result = kde_1d.kds
    grid_expected = grid_expected_1d

    assert np.allclose(grid_result, grid_expected)


def test_1d_kde_df_dindex():
    kde_1d = KernelDensityLines(
        y_res=y_res, kde_kernel="linear", bandwidth=bandwidth
    )

    kde_1d.fit(y_lines=ys_df_di, x=None)

    grid_result = kde_1d.kds
    grid_expected = grid_expected_1d

    assert np.allclose(grid_result, grid_expected)


def test_1d_kde_df_strindex():
    kde_1d = KernelDensityLines(
        y_res=y_res, kde_kernel="linear", bandwidth=bandwidth
    )

    kde_1d.fit(y_lines=ys_df_stri, x=None)

    grid_result = kde_1d.kds
    grid_expected = grid_expected_1d

    assert np.allclose(grid_result, grid_expected)


def test_2d_kde():
    kde_2d = KernelDensityLines(
        y_res=y_res, kde_kernel="linear", bandwidth=bandwidth, dimensions=2
    )
    kde_2d.fit(y_lines=ys, x=x)

    grid_result = kde_2d.kds
    grid_expected = grid_expected_2d

    assert np.allclose(grid_result, grid_expected)


def test_2d_kde_df():
    kde_2d = KernelDensityLines(
        y_res=y_res, kde_kernel="linear", bandwidth=bandwidth, dimensions=2
    )
    kde_2d.fit(y_lines=ys_df, x=None)

    grid_result = kde_2d.kds
    grid_expected = grid_expected_2d

    assert np.allclose(grid_result, grid_expected)


def test_kd_at_y():
    kde_1d = KernelDensityLines(
        y_res=y_res, kde_kernel="linear", bandwidth=bandwidth
    )
    kde_1d.fit(y_lines=ys, x=x)

    result = kde_1d.kd_at_y(y=12.3)
    expected = kd_at_y_1d_expected

    assert np.allclose(result, expected)


def test_kde_scipy_1d_scott_bw():
    # scipy_kernel.factor should be the bandwidth factor (scott default)
    # however, the bandwidth scipy uses multiplies this factor with std of the sample
    # https://stackoverflow.com/questions/23630515/getting-bandwidth-used-by-scipys-gaussian-kde-function

    bw = KernelDensityLines.scott_bandwidth(y_n, 1)
    # scipy_kernel.covariance_factor() * np.std(ys[:,0]) # same as bw*np.std(ys[:,0])

    bw_scipy = bw / np.std(ys[:, 0])
    scipy_kernel = stats.gaussian_kde(ys[:, 0], bw_method=bw_scipy)

    kde_1d = KernelDensityLines(
        y_res=y_res,
        kde_kernel="gaussian",
        bandwidth="scott",
        scale=False,
    )
    kde_1d.fit(y_lines=ys[:, 0:2], x=np.arange(2))
    kdlines_e = kde_1d.kds[0]

    y_eval = np.linspace(ys[:, 0:2].min(), ys[:, 0:2].max(), y_res)
    scipy_e = scipy_kernel.evaluate(y_eval)

    assert np.allclose(scipy_e, kdlines_e[::-1], atol=0.001)


def test_kde_scipy_1d_silverman_bw():
    # scipy_kernel.factor should be the bandwidth factor (scott default)
    # however, the bandwidth scipy uses multiplies this factor with std of the sample
    # https://stackoverflow.com/questions/23630515/getting-bandwidth-used-by-scipys-gaussian-kde-function

    bw = KernelDensityLines.silverman_bandwidth(y_n, 1)
    # scipy_kernel.covariance_factor() * np.std(ys[:,0]) # same as bw*np.std(ys[:,0])

    bw_scipy = bw / np.std(ys[:, 0])
    scipy_kernel = stats.gaussian_kde(ys[:, 0], bw_method=bw_scipy)

    kde_1d = KernelDensityLines(
        y_res=y_res,
        kde_kernel="gaussian",
        bandwidth="silverman",
        scale=False,
    )
    kde_1d.fit(y_lines=ys[:, 0:2], x=np.arange(2))
    kdlines_e = kde_1d.kds[0]

    y_eval = np.linspace(ys[:, 0:2].min(), ys[:, 0:2].max(), y_res)
    scipy_e = scipy_kernel.evaluate(y_eval)

    assert np.allclose(scipy_e, kdlines_e[::-1], atol=0.001)


def test_bw_scott():
    bw = KernelDensityLines.scott_bandwidth(1000, 1)
    assert bw == pytest.approx(0.251188, 0.0001)

    bw2 = KernelDensityLines.scott_bandwidth(1000, 2)
    assert bw2 == pytest.approx(0.316227, 0.0001)

    # compare to scipy,1d and 2d
    scipy_kernel = stats.gaussian_kde(ys[:, 0])
    assert bw == pytest.approx(scipy_kernel.factor, 0.0001)
    # need different values for scipy kde decomposition to pass in 2d
    scipy_kernel = stats.gaussian_kde(values_2d)
    assert bw2 == pytest.approx(scipy_kernel.factor, 0.0001)


def test_bw_silverman():
    bw = KernelDensityLines.silverman_bandwidth(1000, 1)
    assert bw == pytest.approx(0.266065, 0.0001)

    bw2 = KernelDensityLines.silverman_bandwidth(1000, 2)
    assert bw2 == pytest.approx(0.316227, 0.0001)

    # compare to scipy,1d and 2d
    scipy_kernel = stats.gaussian_kde(ys[:, 0], bw_method="silverman")
    assert bw == pytest.approx(scipy_kernel.factor, 0.0001)
    # need different values for scipy kde decomposition to pass in 2d
    scipy_kernel = stats.gaussian_kde(values_2d)
    assert bw2 == pytest.approx(scipy_kernel.factor, 0.0001)
