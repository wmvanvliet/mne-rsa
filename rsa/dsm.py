# encoding: utf-8
"""
Methods to compute dissimilarity matrices (DSMs).
"""

import numpy as np
from scipy.spatial import distance

from .folds import _create_folds


def compute_dsm(data, metric='correlation', **kwargs):
    """Compute a dissimilarity matrix (DSM).

    Parameters
    ----------
    data : ndarray, shape (n_items, ...)
        For each item, all the features. The first are the items and all other
        dimensions will be flattened and treated as features.
    metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'
        (=Pearson correlation).
    **kwargs : dict, optional
        Extra arguments for the distance metric. Refer to
        :mod:`scipy.spatial.distance` for a list of all other metrics and their
        arguments.

    Returns
    -------
    dsm : ndarray, shape (n_classes * n_classes-1,)
        The DSM, in condensed form.
        See :func:`scipy.spatial.distance.squareform`.

    See Also
    --------
    compute_dsm_cv
    """
    X = np.reshape(data, (len(data), -1))
    n_items, n_features = X.shape

    # Be careful with certain metrics
    if n_features == 1 and metric in ['correlation', 'cosine']:
        raise ValueError("There is only a single feature, so "
                         "'correlataion' and 'cosine' can not be "
                         "used as DSM metric. Consider using 'sqeuclidean' "
                         "instead.")

    return distance.pdist(X, metric, **kwargs)


def compute_dsm_cv(folds, metric='correlation', **kwargs):
    """Compute a dissimilarity matrix (DSM) using cross-validation.

    The distance computation is performed from the average of
    all-but-one "training" folds to the remaining "test" fold. This is repeated
    with each fold acting as the "test" fold once. The resulting distances are
    averaged and the result used in the final DSM.

    Parameters
    ----------
    folds : ndarray, shape (n_folds, n_items, ...)
        For each item, all the features. The first dimension are the folds used
        for cross-validation, items are along the second dimension, and all
        other dimensions will be flattened and treated as features.
    metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'
        (=Pearson correlation).
    **kwargs : dict, optional
        Extra arguments for the distance metric. Refer to
        :mod:`scipy.spatial.distance` for a list of all other metrics and their
        arguments.

    Returns
    -------
    dsm : ndarray, shape (n_classes * n_classes-1,)
        The cross-validated DSM, in condensed form.
        See :func:`scipy.spatial.distance.squareform`.

    See Also
    --------
    compute_dsm
    """
    X = np.reshape(folds, (folds.shape[0], folds.shape[1], -1))
    n_folds, n_items, n_features = X.shape[:3]

    # Be careful with certain metrics
    if n_features == 1 and metric in ['correlation', 'cosine']:
        raise ValueError("There is only a single feature, so "
                         "'correlataion' and 'cosine' can not be "
                         "used as DSM metric. Consider using 'sqeuclidean' "
                         "instead.")

    dsm = np.zeros((n_items * (n_items - 1)) // 2)

    X_mean = X.mean(axis=0)

    # Do cross-validation
    for test_fold in range(n_folds):
        X_test = X[test_fold]
        X_train = X_mean - (X_mean - X_test) / (n_folds - 1)

        dist = distance.cdist(X_train, X_test, metric, **kwargs)
        dsm += dist[np.triu_indices_from(dist, 1)]

    return dsm / n_folds


def _ensure_condensed(dsm, var_name):
    """Converts a DSM to condensed form if needed."""
    if type(dsm) is list:
        return [_ensure_condensed(d, var_name) for d in dsm]

    if dsm.ndim == 2:
        if dsm.shape[0] != dsm.shape[1]:
            raise ValueError('Invalid dimensions for "%s". The DSM should '
                             'either be a square matrix, or a one dimensional '
                             'array when in condensed form.')
        dsm = distance.squareform(dsm)
    elif dsm.ndim != 1:
        raise ValueError('Invalid number of dimensions for "%s". The DSM '
                         'should either be a square matrix, or a one '
                         'dimensional array when in condensed form.')
    return dsm


def _n_items_from_dsm(dsm):
    """Get the number of items, given a DSM."""
    if dsm.ndim == 2:
        return dsm.shape[0]
    elif dsm.ndim == 1:
        return distance.squareform(dsm).shape[0]


def dsm_spattemp(data, dist, spatial_radius, temporal_radius,
                 dist_metric='correlation', dist_params=dict(), y=None,
                 n_folds=1, sel_series=None, verbose=False):
    """Generator of DSMs using a spatio-temporal searchlight pattern.

    For each series (i.e. channel or vertex), each time sample is visited. For
    each visited sample, a DSM is computed using the samples within the
    surrounding time region (``temporal_radius``) on all series within the
    surrounding spatial region (``spatial_radius``).

    The data can contain duplicate recordings of the same item (``y``), in
    which case the duplicates will be averaged together before being presented
    to the distance function.

    When performing cross-validation (``n_folds``), duplicates will be split
    into folds. The distance computation is performed from the average of
    all-but-one "training" folds to the remaining "test" fold. This is repeated
    with each fold acting as the "test" fold once. The resulting distances are
    averaged and the result used in the final DSM.

    Parameters
    ----------
    data : ndarray, shape (n_items, n_series, n_times, ...)
        The brain data in the form of an ndarray. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis. The third dimension should contain samples in
        time. Any further dimensions beyond the third will be flattened into a
        single vector.
    dist : ndarray, shape (n_series, n_series)
        The distances between all source points or sensors in meters.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters.
    temporal_radius : int
        The temporal radius of the searchlight patch in samples.
    dist_metric : str
        The distance metric to use to compute the DSMs. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. See also the
        ``dist_params`` parameter to specify and additional parameter for the
        distance function. Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs.
        When ``None``, each item is assumed to belong to a different class.
        Defaults to ``None``.
    n_folds : int
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify -1 to
        use the maximum number of folds possible, given the data.
        Defaults to 1 (no cross-validation).
    sel_series : ndarray, shape (n_selected_series,) | None
        When set, the analysis will be restricted to the subset of time series
        with the given indices. Defaults to ``None``, in which case all series
        are processed.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to ``False``.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each spatio-temporal patch.

    See Also
    --------
    compute_dsm
    dsm_spat
    dsm_temp
    """
    n_series, n_samples = data.shape[1:]

    # Create folds for cross-validated DSM metrics
    folds = _create_folds(data, y, n_folds)
    # The data is now folds x items x n_series x n_times

    centers = _get_time_patch_centers(n_samples, temporal_radius)

    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=n_series * len(centers))

    # Iteratore over spatio-temporal searchlight patches
    if sel_series is None:
        sel_series = range(n_series)
    for series in sel_series:
        for sample in centers:
            patch = folds[
                :, :,
                np.flatnonzero(dist[series] < spatial_radius), ...,
                sample - temporal_radius:sample + temporal_radius
            ]
            if len(folds) == 1:
                dsm_func = compute_dsm
            else:
                dsm_func = compute_dsm_cv
            yield dsm_func(patch, dist_metric, **dist_params)
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()


def dsm_spat(data, dist, spatial_radius, dist_metric='correlation',
             dist_params=dict(), y=None, n_folds=1, sel_series=None,
             verbose=False):
    """Generator of DSMs using a spatial searchlight pattern.

    Each time series (i.e. channel or vertex) is visited. For each visited
    series, a DSM is computed using the entire timecourse of all series within
    the surrounding spatial region (``spatial_radius``).

    The data can contain duplicate recordings of the same item (``y``), in
    which case the duplicates will be averaged together before being presented
    to the distance function.

    When performing cross-validation (``n_folds``), duplicates will be split
    into folds. The distance computation is performed from the average of
    all-but-one "training" folds to the remaining "test" fold. This is repeated
    with each fold acting as the "test" fold once. The resulting distances are
    averaged and the result used in the final DSM.

    Parameters
    ----------
    data : ndarray, shape (n_items, n_series, ..)
        The brain data in the form of an ndarray. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis. All dimensions beyond the second will be
        flattened into a single vector.
    dist : ndarray, shape (n_series, n_series)
        The distances between all source points or sensors in meters.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters.
    dist_metric : str
        The distance metric to use to compute the DSMs. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. See also the
        ``dist_params`` parameter to specify and additional parameter for the
        distance function. Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs.
        When ``None``, each item is assumed to belong to a different class.
        Defaults to ``None``.
    n_folds : int
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify -1 to
        use the maximum number of folds possible, given the data.
        Defaults to 1 (no cross-validation).
    sel_series : ndarray, shape (n_selected_series, ) | None
        When set, the analysis will be restricted to the subset of time series
        with the given indices. Defaults to ``None``, in which case all series
        are processed.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each spatial patch.

    See Also
    --------
    compute_dsm
    dsm_temp
    dsm_spattemp
    """
    # Create folds for cross-validated DSM metrics
    folds = _create_folds(data, y, n_folds)
    # The data is now folds x items x n_series x ...

    if len(folds) == 1:
        dsm_func = compute_dsm
    else:
        dsm_func = compute_dsm_cv

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=len(dist))

    # Iterate over time series
    if sel_series is not None:
        dist = dist[sel_series]
    for series_dist in dist:
        # Construct a searchlight patch of the given radius
        patch = folds[:, :, np.flatnonzero(series_dist < spatial_radius)]
        yield dsm_func(patch, dist_metric, **dist_params)
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()


def dsm_temp(data, temporal_radius, dist_metric='correlation',
             dist_params=dict(), y=None, n_folds=1, sel_times=None,
             verbose=False):
    """Generator of DSMs using a searchlight in time.

    Each time sample is visited. For each visited sample, a DSM is computed
    using the samples within the surrounding time region (``temporal_radius``)
    of all time series (i.e. channels or vertices).

    The data can contain duplicate recordings of the same item (``y``), in
    which case the duplicates will be averaged together before being presented
    to the distance function.

    When performing cross-validation (``n_folds``), duplicates will be split
    into folds. The distance computation is performed from the average of
    all-but-one "training" folds to the remaining "test" fold. This is repeated
    with each fold acting as the "test" fold once. The resulting distances are
    averaged and the result used in the final DSM.

    Parameters
    ----------
    data : ndarray, shape (n_items, n_series, ..., n_times)
        The brain data in the form of an ndarray. The last dimension should be
        consecutive time samples. All dimensions between the second (n_series)
        and final (n_times) dimesions will be flattened into a single vector.
    dist : ndarray, shape (n_series, n_series)
        The distances between all source points or sensors in meters.
    temporal_radius : float
        The temporal radius of the searchlight patch in samples.
    dist_metric : str
        The distance metric to use to compute the DSMs. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. See also the
        ``dist_params`` parameter to specify and additional parameter for the
        distance function. Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs.
        When ``None``, each item is assumed to belong to a different class.
        Defaults to ``None``.
    n_folds : int
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify -1 to
        use the maximum number of folds possible, given the data.
        Defaults to 1 (no cross-validation).
    sel_times : ndarray, shape (n_selected_times,) | None
        When set, the analysis will be restricted to the subset of time points
        with the given indices. Defaults to ``None``, in which case all time
        points are processed.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Yields
    ------
    dsm : ndarray, shape (chunk_size, n_items, n_items)
        A DSM for each time sample

    See Also
    --------
    compute_dsm
    dsm_spat
    dsm_spattemp
    """
    n_samples = data.shape[-1]

    # Iterate over time
    centers = _get_time_patch_centers(n_samples, temporal_radius)

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=len(centers))

    # Create folds for cross-validated DSM metrics
    folds = _create_folds(data, y, n_folds)
    # The data is now folds x items x ... x n_samples

    if len(folds) == 1:
        dsm_func = compute_dsm
    else:
        dsm_func = compute_dsm_cv

    for center in centers:
        # Construct a searchlight patch of the given radius
        patch = folds[..., center - temporal_radius:center + temporal_radius]
        yield dsm_func(patch, dist_metric, **dist_params)
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()


def _get_time_patch_centers(n_samples, temporal_radius):
    """Compute the centers for the temporal patches.

    Parameters
    ----------
    n_samples : int
        The total number of samples.
    temporal_radius : int
        The temporal radius of the patches in samples.

    Returns
    -------
    time_inds : list of int
        For each temporal patch, the time-index of the middle of the patch
    """
    return list(range(temporal_radius, n_samples - temporal_radius + 1))


def _sliding_window(X, win_size):
    """Expose a sliding window view of a numpy array along the last axis.

    This is achieved by manipulating the strides of the numpy array, so data
    does not need to be copied. An new dimension is prepended to the existing
    dimensions, containing the windows. The offset between two consecutive
    windows is 1.

    Parameters
    ----------
    X : ndarray, shape (..., time)
        The numpy array to compute sliding windows on.
    win_size : int
        The size of the window.

    Returns
    -------
    windows : ndarray, shape (n_windows, ..., win_size)
        The numpy array with an added dimension. Selecting along this dimension
        selects sliding windows.

    Examples
    --------
    >>> A = np.arange(5)
    >>> _sliding_window(A, 3)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

    Notes
    -----
    If the window size is larger than the last dimension of the numpy array, no
    windows can be created and an empty array is returned (i.e the length of
    the first dimension of the returned array will be 0).
    """
    n_windows = max(0, 1 + X.shape[-1] - win_size)
    new_shape = (n_windows, *X.shape[:-1], win_size)
    new_strides = (X.strides[-1], *X.strides[:-1], X.strides[-1])
    return np.ndarray(new_shape, dtype=X.dtype, buffer=X, strides=new_strides)
