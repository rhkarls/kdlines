# -*- coding: utf-8 -*-
"""Kernel density estimation and heatmap visualisation on many lines,
time series or 2D data using scikit-learn.

License: MIT (see LICENSE file)
Author: Reinert Huseby Karlsen, copyright 2022.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.colors import Colormap

from typing import Union, Optional

from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt


class KernelDensityLines:
    """Kernel density estimation and heatmap visualisation on many lines,
    time series or 2D data using scikit-learn.
    
    Parameters
    ----------
    kde_kernel : str, optional
        The kernel to use when calling sklearn.neighbors.KernelDensity. 
        Alternatives are {‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, 
                          ‘linear’, ‘cosine’}. 
        See Notes below. The default is 'linear'.
    bandwidth : str or float, optional
        The bandwidth of the kernel, either as float value or method to calculate
        the bandwidth value. Available methods are {'scott', 'silverman'}.
        See Notes below.
        The default is 'scott'.
    y_res : int, optional
        Resolution used on the y axis of the heatmap. The default is 100.
    x_res : int, optional
        Resolution used on the x axis of the heatmap. When set to None, the
        resolution is equal to the length of each y_line. The default is None.
    scale : bool, optional
        Scaling of the resulting kernel density estimation to values between 0 and 1.
        The default is True.
    dimensions : int, optional
        The dimensions to estimate the kernel density over. Can be either 1 or 2.
        When set to 1, the KDEs is done separetly on each x value (or each y 
        lines index if no x is passed). This is recommended if all ys have data 
        at the same x or index. If set to 2, the KDE is done once for all the 
        data at the same time - this is slower and results in similar result dimensions=1
        for data that with same x. If y_lines passed to fit() method has data
        on different x-values/index using dimensions=2 is recommended.
        See Notes below. The default is 1.        
    **kde_kwargs : dict
        Additional kwargs to pass to sklearn.neighbors.KernelDensity.
        Note: Currently not implemented

    Returns
    -------
    KernelDensityLines instance
    
    Notes
    _____
    
    Note that the **choice of kernel** can have a large effect on the speed of the KDE.
    
    **Selection of bandwidth** has a large influence the estimate from the KDE, often
    much more than the kernel type used. The provided methods, 'scott' and 'silverman',
    work well for data that is close to normal and unimodal, but not for other cases.

    The **dimensions keyword**, which can be either 1 or 2, can have a large effect on the
    speed of the KDE, while results *can* be practically identical. Using dimensions=2
    can slow down a factor of 10x or more, and is especially slow if using a more complicated
    kernel (e.g. gaussian compared to linear).
    """

    def __init__(
            self,
            kde_kernel: Optional[str] = "linear",
            bandwidth: Optional[Union[str, float]] = "scott",
            y_res: Optional[int] = 100,
            x_res: Optional[int] = None,
            scale: Optional[bool] = True,
            dimensions: Optional[int] = 1,
            **kde_kwargs,
    ) -> None:

        # variables from arguments
        self.kde_kernel = kde_kernel
        self.y_res = y_res
        self.x_res = x_res
        self.scale = scale
        self.dimensions = dimensions
        self.kde_kwargs = kde_kwargs

        # variables for fitting, data, plotting
        self.kds = None
        self.xygrid = None
        self.x_plot = None
        self.y_plot = None
        self.x_plot_labels = None

        # set bandwidth function
        if bandwidth == "scott":
            self.bw_fv = self.scott_bandwidth
        elif bandwidth == "silverman":
            self.bw_fv = self.silverman_bandwidth
        else:
            self.bw_fv = bandwidth

    def fit(
            self,
            y_lines: Union[np.ndarray, pd.DataFrame],
            x: Optional[np.ndarray] = None,
    ):
        """
        Fit the kernel density model with the provided data.

        The fitted density is stored under attribute `kds`.

        Parameters
        ----------
        y_lines : numpy.ndarray or pandas.DataFrame
            shape requirements for the nd.array.
        x : numpy.ndarray, optional
            what happens when None with y being array or dataframe
            shape requirement. The default is None.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # if user passed pandas dataframe, take the index as x and colums as ylines
        if hasattr(y_lines, "to_numpy"):
            try:
                if x is None:
                    x = y_lines.index.to_numpy()
                # transpose the dataframe to same format as numpy arrays
                y_lines = y_lines.T.to_numpy()
            except (AttributeError, TypeError):
                print(
                    "Please pass numpy arrays or pandas DataFrame to fit() method."
                )
                raise

        if x is None:
            x = np.arange(len(y_lines[0]))

        if self.x_res is None:
            self.x_res = len(x)

        # check x if numeric, so that grid and plots can be created from it
        # in any case, keep original x for plot labels
        self.x_plot_labels = x.copy()
        if not np.issubdtype(x.dtype, np.number):
            x = np.arange(len(x))

        if callable(self.bw_fv):
            kde_bw = self.bw_fv(len(y_lines), self.dimensions)
        else:
            kde_bw = self.bw_fv

        self.x_plot = x
        self.y_plot = y_lines

        # Setup grid
        # noinspection PyArgumentList
        x_min = x.min()
        # noinspection PyArgumentList
        x_max = x.max()
        y_min = y_lines.min()
        y_max = y_lines.max()

        grid_size_x = self.x_res
        grid_size_y = self.y_res
        # Generate grid
        xygrid = np.mgrid[
                 x_min: x_max: complex(grid_size_x),
                 y_min: y_max: complex(grid_size_y),
                 ]

        self.xygrid = xygrid

        grid_x = xygrid[0]
        grid_y = xygrid[1]

        if self.dimensions == 1:
            kds = self._calc_kde_1d(x, y_lines, grid_x, grid_y, kde_bw)
        elif self.dimensions == 2:
            kds = self._calc_kde_2d(x, y_lines, grid_x, grid_y, kde_bw)
        else:
            raise ValueError(f"Invalid value for dimensions keyword ({self.dimensions})."
                             "Must be either 1 or 2.")

        if self.scale:
            # noinspection PyArgumentList
            kds = (kds - kds.min()) / (kds.max() - kds.min())

        self.kds = kds

        return self

    def _calc_kde_1d(
            self,
            x: np.ndarray,
            y_lines: np.ndarray,
            grid_x: np.ndarray,
            grid_y: np.ndarray,
            kde_bw: float,
    ) -> np.ndarray:
        """KDE on 1d arrays."""
        kds = np.empty((grid_x.shape[0], grid_y.shape[1]))
        # loop over each x and fit KD model to y values at x
        for i, _ in enumerate(x):
            kde = KernelDensity(kernel=self.kde_kernel, bandwidth=kde_bw).fit(
                y_lines[:, i].reshape(1, -1).T
            )
            kde_s_log = kde.score_samples(
                grid_y[0].reshape(-1, 1)[::-1]  # reversed, Y is flipped on grid
            )
            kde_s = np.exp(kde_s_log)
            kds[i] = kde_s

        return kds

    def _calc_kde_2d(
            self,
            x: np.ndarray,
            y_lines: np.ndarray,
            grid_x: np.ndarray,
            grid_y: np.ndarray,
            kde_bw: float,
    ) -> np.ndarray:
        """KDE on 2d array."""
        # tile x array to same shape as ys
        xs = np.broadcast_to(x, (y_lines.shape[0], y_lines.shape[1]))
        # format to x, y shape
        xy = np.vstack([xs.ravel(), y_lines.ravel()]).T

        # for sampling the kde
        grid_samples = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

        kde = KernelDensity(kernel=self.kde_kernel, bandwidth=kde_bw)
        kde_fit = kde.fit(xy)

        kde_s_log = kde_fit.score_samples(grid_samples)
        kds = np.exp(kde_s_log).reshape(grid_x.shape)

        kds = np.fliplr(kds)  # reversed, Y is flipped on grid

        return kds

    def plot(
            self,
            ax: Optional[plt.Axes] = None,
            cmap: Optional[Union[str, Colormap]] = "viridis",
            aspect: Optional[Union[str, float]] = "auto",
            show_data: Optional[bool] = False,
            time_label_format: Optional[str] = "%Y-%m-%d",
            **kwargs,
    ) -> plt.Axes:
        """
        Plot the fitted density as a heatmap.

        Parameters
        ----------
        ax : matplotlib axes object, optional
            Axes to plot the heatmap on. If None a new figure and axes
            is created. The default is None.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap used for density heatmap.
            See https://matplotlib.org/stable/tutorials/colors/colormaps.html
            The default is 'viridis'.
        aspect : str or float, optional
            Aspect ration of the axes, see matplotlib.pyplot.imshow.
            The default is 'auto'.
        show_data : bool, optional
            Plot the data that the kde is based on.
            Plotted as black thin lines. The default is False.
        time_label_format : str, optional
            If x values are datetime/timestamps format to this string
            on the plot labels. See datetime.strftime
            The default is '%Y-%m-%d'.
        **kwargs : dict
            Additional keyword arguments for matplotlib.pyplot.imshow.

        Returns
        -------
        ax : matplotlib axes object
            The axes the heatmap is plotted on.

        """
        if self.kds is None:
            raise NotFittedError("Nothing to plot. ")

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(
            self.kds.T,
            cmap=cmap,
            extent=[
                self.x_plot.min(),
                self.x_plot.max(),
                self.y_plot.min(),
                self.y_plot.max(),
            ],
            aspect=aspect,
            **kwargs,
        )

        if show_data:
            ax.plot(self.x_plot, self.y_plot.T, color="k", alpha=0.5, lw=0.5)

        # set x tick labels when x data is not numeric
        if not np.issubdtype(self.x_plot_labels.dtype, np.number):
            x_tick_locs = ax.get_xticks()
            x_tick_locs = x_tick_locs.astype(int)
            try:
                x_tick_labels = self.x_plot_labels[x_tick_locs]
            except IndexError:  # often the returned ticks are just outside of the range
                x_tick_locs = x_tick_locs[:-1]
                x_tick_labels = self.x_plot_labels[x_tick_locs]

            # use the time_label_format if datetime ticks
            # casting to datetime using item(), but needs to be as [s] then
            if np.issubdtype(x_tick_labels.dtype, np.datetime64):
                new_x_tick_labels = []
                for i, l in enumerate(x_tick_labels):
                    new_x_tick_labels.append(
                        l.astype("datetime64[s]")
                        .item()
                        .strftime(time_label_format)
                    )
            else:
                new_x_tick_labels = x_tick_labels

            ax.set_xticks(x_tick_locs)
            ax.set_xticklabels(new_x_tick_labels)

        return ax

    def kd_at_y(self, y: float) -> np.ndarray:
        """
        Extract density values at y.

        Parameters
        ----------
        y : float
            DESCRIPTION.

        Returns
        -------
        np.ndarray
            DESCRIPTION.

        """
        if self.kds is None:
            raise NotFittedError

        grid_row_e = np.argmin(abs(self.xygrid[1][0, :] - y))
        # kds flipped on y axis relative to grid
        return self.kds[:, self.y_res - grid_row_e]

    @staticmethod
    def silverman_bandwidth(n: int, d: int) -> float:
        """Calculate bandwidth using Silverman method.

        Parameters
        ----------
        n : int
            number of data points
        d : int
            dimensions of the data.


        Returns
        -------
        float
            KDE bandwidth according to Silverman

        Silverman, B. W. 1992. Density Estimation for Statistics and Data Analysis.
        London: Chapman & Hall. ISBN 9780412246203

        """
        return (n * (d + 2) / 4.0) ** (-1 / (d + 4))

    @staticmethod
    def scott_bandwidth(n: int, d: int) -> float:
        """Calculate bandwidth using Scott method.

        Parameters
        ----------
        n : int
            number of data points
        d : int
            dimensions of the data.


        Returns
        -------
        float
            KDE bandwidth according to Scott

        Scott, D. W. (1992). Multivariate Density Estimation: Theory, Practice,
        and Visualization. New York: Wiley

        """
        return n ** (-1.0 / (d + 4))


class NotFittedError(Exception):
    """Exception for not fitted error.

    Attributes
    ----------
    message : str
        error message
    """

    def __init__(self, message=""):
        self.message = message

    def __str__(self):
        return (
            f"{self.message}This KernelDensityLines instance is not yet fitted. "
            f"Call fit() before using this method."
        )
