from functools import partial
import types
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mne.viz import Brain
from mne.viz.topo import _iter_topography
import numpy as np
from scipy.spatial import distance


def plot_dsms(dsms, names=None, items=None, n_rows=1, cmap='viridis',
              title=None):
    """Plot one or more DSMs

    Parameters
    ----------
    dsms : ndarray | list of ndarray
        The DSM or list of DSMs to plot. The DSMs can either be two-dimensional
        (n_items x n_items) matrices or be in condensed form.
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
            i = row * n_cols + col % n_cols
            if i < len(dsms):
                dsm = dsms[i]
                if dsm.ndim == 1:
                    dsm = distance.squareform(dsm)
                elif dsm.ndim > 2:
                    raise ValueError(f'Invalid shape {dsm.shape} for DSM')
                im = ax[row, col].imshow(dsm, cmap=cmap)

                if names is not None:
                    name = names[i]
                    ax[row, col].set_title(name)
                if items is not None:
                    ax[row, col].set_xticks(np.arange(len(items)))
                    ax[row, col].set_xticklabels(items)
                    ax[row, col].set_yticks(np.arange(len(items)))
                    ax[row, col].set_yticklabels(items)
            else:
                ax[row, col].set_visible(False)

    plt.colorbar(im, ax=ax)
    if title is not None:
        plt.suptitle(title)
    return fig


def _click_func(ax, ch_idx, dsms, cmap):
    """ Function used to plot a single DSM interactively.

    Parameters
    ----------
    ax: matplotlib.Axes.axes
        Axes.axes object on which a new single DSM is plotted.
    ch_idx: int
        Index of a channel.
    dsms: ndarray, shape (n_sensors, n_dsm_datapoint)
        DSMs of MEG recordings; there's one DSM for each sensor.
    cmap: str
        Colormap used for plotting DSMs.
        Check matplotlib.pyplot.imshow for details.
    """
    dsm = dsms[ch_idx]
    dsm = distance.squareform(dsm)
    ax.imshow(dsm, cmap=cmap)


def _plot_dsms_topo_timepoint(dsms, info, layout=None, fig=None, title=None,
                              axis_facecolor='w', axis_spinecolor='w',
                              fig_facecolor='w', figsize=(6.4, 4.8),
                              cmap='viridis', show=False):
    """Plot DSMs on 2D MEG topography.

    Parameters
    ----------
    dsms: ndarray, shape (n_sensors, n_dsm_datapoints) | generator
        DSMs of MEG recordings; one DSM for each sensor.
        Can also be a generator of DSMs as produced by the :func:`dsm_epochs`,
        :func:`dsm_evokeds` or :func:`dsm_array` functions.
    info: mne.io.meas_info.Info
        Info object that contains meta data of MEG recordings.
    layout: mne.channels.layout.Layout | None
        Layout objects containing sensor layout info.
        The default (``None``) will figure out layout based on info.
    fig: matplotlib.pyplot.Figure | None
        Figure object on which DSMs on 2D MEG topography are plotted.
        The default (``None``) creates a new Figure object.
    title: str | None
        Title of the plot, used only when ``fig=None``.
        The default (``None``) puts no title in the figure.
    axis_facecolor: str
        Face color of the each DSM. Defaults to 'w', white.
    axis_spinecolor: str
        Spine color of each DSM. Defaults to 'w', white.
    fig_facecolor: str
        Face color of the entire topography. Defaults to 'w', white.
    figsize: tuple of float
        Figure size. The first element specify width and the second height.
        Defaults to (6.4, 4.8).
    cmap: str
        Colormap used for plotting DSMs. Defaults to 'viridis'.
        Check :func:`matplotlib.pyplot.imshow` for details.
    show: bool
        Whether to display the generated figure. Defaults to False.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Figure object in which DSMs are plotted on 2D MEG topography.
    """
    on_pick = partial(_click_func, dsms=dsms, cmap=cmap)

    if fig is None:
        fig = plt.figure(figsize=figsize)
        if title is not None:
            fig.suptitle(title, x=0.98, horizontalalignment='right')
    else:
        fig = plt.figure(fig.number)

    my_topo_plot = _iter_topography(info=info, layout=layout, on_pick=on_pick,
                                    fig=fig, axis_facecolor=axis_facecolor,
                                    axis_spinecolor=axis_spinecolor,
                                    fig_facecolor=fig_facecolor,
                                    unified=False)

    for i, (ax, _) in enumerate(my_topo_plot):
        dsms_i = dsms[i]
        dsms_i = distance.squareform(dsms_i)
        ax.imshow(dsms_i, cmap=cmap)

    if show:
        fig.show()

    return fig


def plot_dsms_topo(dsms, info, time=None, layout=None, fig=None,
                   axis_facecolor='w', axis_spinecolor='w', fig_facecolor='w',
                   figsize=(6.4, 4.8), cmap='viridis', show=True):
    """ Plot DSMs on 2D sensor topography

    Parameters
    ----------
    dsms: ndarray | numpy.memmap, shape (n_sensors,[ n_times,] n_dsm_datapts)
        DSMs of MEG/EEG recordings; one DSM for each sensor and time point.
    info: mne.io.meas_info.Info
        Info object that contains meta data of MEG/EEG recordings.
    time: int | [int, int] | None
        A time point (int) or time window ([int, int]) for which DSMs are
        plotted. When a time window is given, averge DSMs for the window are
        plotted. The default (``None``) plots the average DSMs of all the time
        points. Start of the time window is inclusive, while the end is
        exclusive.
    layout: mne.channels.layout.Layout, optional
        Layout objects containing sensor layout info.
        The default, ``layout=None``, will figure out layout based on info.
    fig: matplotlib.pyplot.Figure | None, optional
        Figure object on which DSMs on 2D sensor topography are plotted.
        The default (``None``) creates a new Figure object
        with a title based on time parameter.
    axis_facecolor: str, optional
        Face color of the each DSM. Defaults to 'w', white.
    axis_spinecolor: str, optional
        Spine color of each DSM. Defaults to 'w', white.
    fig_facecolor: str, optional
        Face color of the entire topography. Defaults to 'w', white.
    figsize: tuple of float, optional
        Figure size. The first element specify width and the second height.
        Defaults to (6.4, 4.8).
    cmap: str, optional
        Colormap used for plotting DSMs. Defaults to 'viridis'.
        Check :func:`matplotlib.pyplot.imshow` for details.
    show: bool, optional
        Whether to display the generated figure. Defaults to ``True``.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        Figure object in which DSMs are plotted on 2D sensor topography.
    """
    if isinstance(dsms, types.GeneratorType):
        dsms = np.array(list(dsms))

    if dsms.ndim != 2 and dsms.ndim != 3:
        raise ValueError('dsms have to be a 2D or 3D ndarray or numpy.memmap, '
                         '[n_sensors,[ n_times,] n_dsm_datapoints]')
    if len(dsms.shape) == 2:
        dsms = dsms[:, np.newaxis, :]
    if time is None:
        time = [0, dsms.shape[1]]
    if isinstance(time, int):
        time = [time, time + 1]
    if not isinstance(time, list):
        raise TypeError('time has to be int, list of [int, int] or None.')
    if (not all(isinstance(i, int) for i in time)) or (len(time) != 2):
        raise TypeError('time has to be int, list of [int, int] or None.')
    if time[0] >= time[1]:
        raise ValueError('The start of the time window has to be smaller '
                         'than the end of the time window.')
    if time[0] < 0 or time[1] > dsms.shape[1]:
        raise ValueError('The time window is out of range. The minimum is 0 '
                         f'and the maximum is {dsms.shape[1]}')
    if (fig is not None) and (not isinstance(fig, plt.Figure)):
        raise TypeError('fig has to be matplotlib.pyplot.Figure or None.')

    dsms_cropped = dsms[:, time[0]:time[1], :]
    dsms_avg = dsms_cropped.mean(axis=1)
    # set title to time window
    if time[0] + 1 != time[1]:
        title = f'From {time[0]} (inclusive) to {time[1]} (exclusive)'
    else:
        title = f'Time point: {time[0]}'

    fig = _plot_dsms_topo_timepoint(dsms_avg, info, fig=fig, layout=layout,
                                    title=title,
                                    axis_facecolor=axis_facecolor,
                                    axis_spinecolor=axis_spinecolor,
                                    fig_facecolor=fig_facecolor,
                                    figsize=figsize, cmap=cmap, show=show)
    return fig


def plot_roi_map(values, rois, subject, subjects_dir, cmap='plasma',
                 alpha=1.0):
    """Plot ROI values on a FreeSurfer brain.

    Parameters
    ----------
    values : array-like, shape (n_rois,)
        The values to plot. One value per ROI.
    rois : list of mne.Label
        The labels corrsponding to the ROIs.
    subject : str
        The name of the FreeSurfer subject to plot the brain for.
    subjects_dir : str
        The folder in which the FreeSurfer subject data is kept. Inside this
        folder should be a folder with the same name as the `subject`
        parameter.
    cmap : str
        The name of the matplotlib colormap to use. Defaults to 'plasma'.
    alpha : float
        The alpha (opacity, 1.0 is fully opaque, 0.0 is fully transparant) of
        the data being plotted on top of the brain.

    Returns
    -------
    brain : mne.viz.Brain
        The MNE-Python brain plotting object that was created and currently
        being shown. You can use this to modify the plot.
    """
    cmap = get_cmap(cmap)
    max_val = np.max(values)
    brain = Brain(subject=subject, subjects_dir=subjects_dir, surf='inflated',
                  hemi='both')
    labels_lh = np.zeros(len(brain.geo['lh'].coords), dtype=int)
    labels_rh = np.zeros(len(brain.geo['rh'].coords), dtype=int)
    ctab_lh = list()
    ctab_rh = list()
    for i, (roi, value) in enumerate(zip(rois, values), 1):
        if roi.hemi == 'lh':
            labels = labels_lh
            ctab = ctab_lh
        else:
            labels = labels_rh
            ctab = ctab_rh
        labels[roi.vertices] = i
        ctab.append([int(x * 255) for x in cmap(value / max_val)[:4]] + [i])
    ctab_lh = np.array(ctab_lh)
    ctab_rh = np.array(ctab_rh)
    brain.add_annotation([(labels_lh, ctab_lh), (labels_rh, ctab_rh)],
                         borders=False, alpha=alpha)
    return brain
