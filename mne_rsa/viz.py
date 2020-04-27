from scipy.spatial import distance
from matplotlib import pyplot as plt
import numpy as np


def plot_dsms(dsms, names=None, items=None, n_rows=1, cmap='viridis',
              title=None):
    """Plot one or more DSMs

    Parameters
    ----------
    dsms : ndarray | list of ndarray
        The DSM or list of DSMs to plot. The DSMs can either be two-dimensional
        (n_items x n_iterms) matrices or be in condensed form.
    names : str | list of str | None
        For each given DSM, a name to show above it. Defaults to no names.
    items : list of str | None
        The each item (row/col) in the DSM, a string description. This will be
        displayed along the axes. Defaults to None which means the items will
        be numbered.
    n_rows : int
        Number of rows to use when plotting multiple DSMs at once. Defaults to
        1.
    cmap : str
        Matplotlib colormap to use. See
        https://matplotlib.org/gallery/color/colormap_reference.html
        for all possibilities. Defaults to 'viridis'.
    title : str | None
        Title for the entire figure. Defaults to no title.

    Returns
    -------
    fig : matplotlib figure
        The figure produced by matplotlib
    """
    if not isinstance(dsms, list):
        dsms = [dsms]

    if isinstance(names, str):
        names = [names]
    if names is not None and len(names) != len(dsms):
        raise ValueError(f'Number of given names ({len(names)}) does not '
                         f'match the number of DSMs ({len(dsms)})')

    n_cols = int(np.ceil(len(dsms) / n_rows))
    fig = plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    ax = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    for row in range(n_rows):
        for col in range(n_cols):
            dsm = dsms[row * n_cols + col % n_cols]
            if dsm.ndim == 1:
                dsm = distance.squareform(dsm)
            elif dsm.ndim > 2:
                raise ValueError(f'Invalid shape {dsm.shape} for DSM')
            im = ax[row, col].imshow(dsm, cmap=cmap)

            if names is not None:
                name = names[row * n_cols + col % n_cols]
                ax[row, col].set_title(name)

    plt.colorbar(im, ax=ax)
    if title is not None:
        plt.suptitle(title)
    return fig
