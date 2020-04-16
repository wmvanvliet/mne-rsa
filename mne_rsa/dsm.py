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
    metric : str | function
        The distance metric to use to compute the DSM. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. When a function is
        specified, it needs to take in two vectors and output a single number.
        See also the ``dist_params`` parameter to specify and additional
        parameter for the distance function.
        Defaults to 'correlation'.
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
    metric : str | function
        The distance metric to use to compute the DSM. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. When a function is
        specified, it needs to take in two vectors and output a single number.
        See also the ``dist_params`` parameter to specify and additional
        parameter for the distance function.
        Defaults to 'correlation'.
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
                 n_folds=1, sel_series=None, sel_times=None, verbose=False):
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
    dist_metric : str | function
        The distance metric to use to compute the DSMs. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. When a function is
        specified, it needs to take in two vectors and output a single number.
        See also the ``dist_params`` parameter to specify and additional
        parameter for the distance function.
        Defaults to 'correlation'.
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
        When set, searchlight patches will only be generated for the subset of
        time series with the given indices. Defaults to ``None``, in which case
        patches for all series are generated.
    sel_times : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time samples with the given indices. Defaults to ``None``, in which
        case patches for all samples are generated.
    verbose : bool | int
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. If an integer value is given, this
        indicates the position of the progress bar (starting from 1, useful for
        showing multiple progress bars at the same time when calling from
        different threads). Defaults to ``False``.

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
    n_series = data.shape[1]
    n_times = data.shape[-1]

    # Create folds for cross-validated DSM metrics
    folds = _create_folds(data, y, n_folds)
    # The data is now folds x items x n_series x n_times

    if sel_series is None:
        sel_series = np.arange(n_series)
    if sel_times is None:
        sel_times = np.arange(n_times)

    # Get a valid time range given the size of the sliding windows
    tmin = max(temporal_radius, sel_times[0])
    tmax = min(n_times - temporal_radius + 1, sel_times[-1])
    centers = np.arange(tmin, tmax)

    if verbose:
        from tqdm import tqdm
        if type(verbose) == int:
            position = verbose - 1
        else:
            position = 0
        pbar = tqdm(total=len(sel_series) * len(centers), position=position)

    # Iterate over spatio-temporal searchlight patches
    for series in sel_series:
        for sample in centers:
            patch = folds[
                :, :,
                np.flatnonzero(dist[series] < spatial_radius), ...,
                sample - temporal_radius:sample + temporal_radius
            ]
            if len(folds) == 1:
                yield compute_dsm(patch[0], dist_metric, **dist_params)
            else:
                yield compute_dsm_cv(patch, dist_metric, **dist_params)
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()


def dsm_spat(data, dist, spatial_radius, dist_metric='correlation',
             dist_params=dict(), y=None, n_folds=1, sel_series=None,
             sel_times=None, verbose=False):
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
    dist_metric : str | function
        The distance metric to use to compute the DSMs. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. When a function is
        specified, it needs to take in two vectors and output a single number.
        See also the ``dist_params`` parameter to specify and additional
        parameter for the distance function.
        Defaults to 'correlation'.
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
        When set, searchlight patches will only be generated for the subset of
        time series with the given indices. Defaults to ``None``, in which case
        patches for all series are generated.
    sel_times : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time samples with the given indices. Defaults to ``None``, in which
        case patches for all samples are generated.
    verbose : bool | int
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. If an integer value is given, this
        indicates the position of the progress bar (starting from 1, useful for
        showing multiple progress bars at the same time when calling from
        different threads). Defaults to ``False``.

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

    if sel_series is not None:
        dist = dist[sel_series]

    # Progress bar
    if verbose:
        from tqdm import tqdm
        if type(verbose) == int:
            position = verbose - 1
        else:
            position = 0
        pbar = tqdm(total=len(dist), position=position)

    # Iterate over time series
    for series_dist in dist:
        # Construct a searchlight patch of the given radius
        patch = folds[:, :, np.flatnonzero(series_dist < spatial_radius)]
        if len(folds) == 1:
            yield compute_dsm(patch[0], dist_metric, **dist_params)
        else:
            yield compute_dsm_cv(patch, dist_metric, **dist_params)
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
    dist_metric : str | function
        The distance metric to use to compute the DSMs. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. When a function is
        specified, it needs to take in two vectors and output a single number.
        See also the ``dist_params`` parameter to specify and additional
        parameter for the distance function.
        Defaults to 'correlation'.
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
    sel_times : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time samples with the given indices. Defaults to ``None``, in which
        case patches for all samples are generated.
    verbose : bool | int
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. If an integer value is given, this
        indicates the position of the progress bar (starting from 1, useful for
        showing multiple progress bars at the same time when calling from
        different threads). Defaults to ``False``.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each time sample

    See Also
    --------
    compute_dsm
    dsm_spat
    dsm_spattemp
    """
    n_times = data.shape[-1]
    if sel_times is None:
        sel_times = np.arange(n_times)

    # Get a valid time range given the size of the sliding windows
    tmin = max(temporal_radius, sel_times[0])
    tmax = min(n_times - temporal_radius + 1, sel_times[-1])
    centers = np.arange(tmin, tmax)

    # Progress bar
    if verbose:
        from tqdm import tqdm
        if type(verbose) == int:
            position = verbose - 1
        else:
            position = 0
        pbar = tqdm(total=len(centers), position=position)

    if n_folds == 1:
        dsm_func = compute_dsm
    else:
        dsm_func = compute_dsm_cv
        # Create folds for cross-validated DSM metrics
        data = _create_folds(data, y, n_folds)
        # The data is now folds x items x ... x n_times

    for center in centers:
        # Construct a searchlight patch of the given radius
        patch = data[..., center - temporal_radius:center + temporal_radius + 1]
        m = dsm_func(patch, dist_metric, **dist_params)
        yield m
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()


def dsm_array(X, dist=None, spatial_radius=None, temporal_radius=None,
              dist_metric='correlation', dist_params=dict(), y=None,
              n_folds=1, sel_series=None, sel_times=None, verbose=False):
    """Generate DSMs from an array of data, possibly in a searchlight pattern.

    This function acts as a ditchpatch: calling the :func:`dsm_spattemp`,
    :func:`dsm_spat`, :func:`dsm_temp` or :func:`comput_dsm` functions as
    necessary, depending on the settings of ``spatial_radius`` and
    ``temporal_radius``.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, n_times)
        An array containing the data.
    dist : ndarray, shape (n_series, n_series) | None
        The distances between all source points or sensors in meters.
        Defaults to ``None``.
    spatial_radius : floats | None
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch. Set to
        None to only perform the searchlight over time, flattening across
        sensors. When this parameter is set, the ``dist`` parameter must also
        be specified. Defaults to ``None``.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in samples. Set to None to
        only perform the searchlight over sensors/vertices, flattening across
        time. Defaults to ``None``.
    dist_metric : str | function
        The distance metric to use to compute the DSMs. Can be any metric
        supported by :func:`scipy.spatial.distance.pdist`. When a function is
        specified, it needs to take in two vectors and output a single number.
        See also the ``dist_params`` parameter to specify and additional
        parameter for the distance function.
        Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs.
        When ``None``, each item is assumed to belong to a different class.
        Defaults to ``None``.
    n_folds : int | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify -1 to
        use the maximum number of folds possible, given the data.
        Defaults to 1 (no cross-validation).
    sel_series : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time series with the given indices. Defaults to ``None``, in which case
        patches for all series are generated.
    sel_times : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time samples with the given indices. Defaults to ``None``, in which
        case patches for all samples are generated.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each searchlight patch.
    """
    # Spatio-temporal searchlight
    if spatial_radius is not None and temporal_radius is not None:
        if dist is None:
            raise ValueError('A spatial radius was requested, but no distance '
                             'information was specified (=dist parameter).')
        yield from dsm_spattemp(
            X, dist, spatial_radius, temporal_radius, dist_metric, dist_params,
            y, n_folds, sel_series, sel_times, verbose=verbose)

    # Spatial searchlight
    elif spatial_radius is not None:
        if dist is None:
            raise ValueError('A spatial radius was requested, but no distance '
                             'information was specified (=dist parameter).')
        yield from dsm_spat(
            X, dist, spatial_radius, dist_metric, dist_params, y, n_folds,
            sel_series, sel_times, verbose=verbose)

    # Temporal searchlight
    elif temporal_radius is not None:
        yield from dsm_temp(
            X, temporal_radius, dist_metric, dist_params, y, n_folds,
            sel_times, verbose=verbose)

    # Only a single searchlight patch
    else:
        folds = _create_folds(X, y, n_folds)
        if len(folds) == 1:
            yield compute_dsm(folds[0], dist_metric, **dist_params)
        else:
            yield compute_dsm_cv(folds, dist_metric, **dist_params)
