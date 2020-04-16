from scipy.spatial import distance
from matplotlib import pyplot as plt
import numpy as np


def plot_dsms(dsms, items=None, n_rows=1):
    """Plot one or more DSMs

    Parameters
    ----------
    dsms : ndarray | list of ndarray
        The DSM or list of DSMs to plot. The DSMs can either be two-dimensional
        (n_items x n_iterms) matrices or be in condensed form.
    items : list of str | None
        The each item (row/col) in the DSM, a string description. This will be
        displayed along the axes. Defaults to None which means the items will
        be numbered.
    n_rows : int
        Number of rows to use when plotting multiple DSMs at once. Defaults to
        1.

    Returns
    -------
    fig : matplotlib figure
        The figure produced by matplotlib
    """
    if not isinstance(dsms, list):
        dsms = [dsms]

    n_cols = int(np.ceil(len(dsms) / n_rows))
    fig = plt.figure(figsize=(2 * n_rows, 2 * n_cols))

    ax = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    for row in range(n_rows):
        for col in range(n_cols):
            dsm = dsms[row * n_cols + col % n_cols]
            if dsm.ndim == 1:
                dsm = distance.squareform(dsm)
            elif dsm.ndim > 2:
                raise ValueError(f'Invalid shape {dsm.shape} for DSM')
            im = ax[row, col].imshow(dsm, cmap='RdBu_r')

    plt.colorbar(im, ax=ax)
    return fig
