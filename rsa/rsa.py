# encoding: utf-8
"""
Module implementing private functions to perform RSA in searchlight patches.


Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
Ossi Lehtonen <ossi.lehtonen@aalto.fi>
"""

import numpy as np
import mne
from scipy import stats
from scipy.spatial import distance


def rsa_spattemp(X, dsm_Y, dist, spatial_radius, temporal_radius,
                 data_dsm_metric='correlation', rsa_metric='spearman',
                 break_after=-1, n_jobs=1, verbose=False):
    """Compute spatio-temporal RSA using a searchlight pattern.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, n_times)
        The brain data in the form of an ndarray. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis.
    dsm_Y : array, shape (n_items, n_items) | (n_items * n_items - 1,)
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`.
    dist : ndarray, shape (n_series, n_series)
        The distances between all source points or sensors in meters.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters.
    temporal_radius : int
        The temporal radius of the searchlight patch in samples.
    data_dsm_metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs, either Spearman or
        Pearson correlation. Defaults to 'spearman'.
    break_after : int
        Abort the computation after this many steps. Useful for debugging. Set
        to -1 to continue until the end. Defaults to -1.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    results : ndarray, shape (n_series, n_times)
        The RSA correlation values for each spatio-temporal patch.
    """
    n_series, n_samples = X.shape[1:]
    dsm_Y = _ensure_condensed(dsm_Y)

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=n_series)

    def progress(sequence):
        step = 0
        for item in sequence:
            if verbose:
                pbar.update(1)
            if break_after >= 0 and step > break_after:
                break
            step += 1
            yield item
        if verbose:
            pbar.close()

    # Use multiple cores to do the RSA computation
    parallel, my_rsa_searchlight_step, _ = mne.parallel.parallel_func(
        rsa_temp, n_jobs, verbose=False)

    # Run RSA for each spatial patch
    results = []
    results += parallel(
        my_rsa_searchlight_step(
            X[:, np.flatnonzero(series_dist < spatial_radius), :],
            dsm_Y, data_dsm_metric, rsa_metric, temporal_radius, verbose=False)
        for series_dist in progress(dist)
    )

    return np.array(results)


def rsa_spat(X, dsm_Y, dist, spatial_radius, data_dsm_metric='correlation',
             rsa_metric='spearman', break_after=-1, n_jobs=1, verbose=False):
    """Compute spatial RSA using a searchlight pattern.

    Computes RSA across space using a searchlight pattern, flattens the
    temporal dimension.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, ...)
        The brain data in the form of an ndarray. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis.
    dsm_Y : array, shape (n_items, n_items) | (n_items * n_items - 1,)
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`.
    dist : ndarray, shape (n_series, n_series)
        The distances between all source points or sensors in meters.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters.
    data_dsm_metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs, either Spearman or
        Pearson correlation. Defaults to 'spearman'.
    break_after : int
        Abort the computation after this many steps. Useful for debugging. Set
        to -1 to continue until the end. Defaults to -1.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    results : ndarray, shape (n_series, n_times)
        The RSA correlation values for each spatio-temporal patch.
    """
    dsm_Y = _ensure_condensed(dsm_Y)
    results = []

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=len(dist))

    # Iterate over space
    for series_dist in dist:
        x = X[:, np.flatnonzero(series_dist < spatial_radius)]
        results.append(_rsa(x, dsm_Y, data_dsm_metric, rsa_metric))
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    return np.array(results)


def rsa_temp(X, dsm_Y, temporal_radius, data_dsm_metric='correlation',
             rsa_metric='spearman', break_after=-1, n_jobs=1, verbose=False):
    """Perform temporal RSA analysis using a searchlight in time.

    Computes RSA across time using a searchlight, flattens the spatial
    dimension.

    Parameters
    ---------
    X : ndarray, shape (n_items, ..., n_times)
        The brain data in the spatial patch. The last dimension should be
        consecutive time points.
    dsm_Y : array, shape (n_items, n_items) | (n_items * n_items - 1,)
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`.
    data_dsm_metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs, either Spearman or
        Pearson correlation. Defaults to 'spearman'.
    break_after : int
        Abort the computation after this many steps. Useful for debugging. Set
        to -1 to continue until the end. Defaults to -1.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    results : ndarray, shape (n_temporal_patches,)
        The RSA correlation values for each temporal patch.
    """
    n_samples = X.shape[-1]
    dsm_Y = _ensure_condensed(dsm_Y)

    # Iterate over time
    results = []

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=n_samples)

    for sample in _get_time_patch_centers(n_samples, temporal_radius):
        x = X[..., sample - temporal_radius:sample + temporal_radius]
        results.append(_rsa(x, dsm_Y, data_dsm_metric, rsa_metric))
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    return np.array(results)


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


def _ensure_condensed(dsm, var_name):
    """Converts a DSM to condensed form if needed."""
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


def _rsa(X, dsm_Y, data_dsm_metric, rsa_metric):
    """Perform RSA between data X and a model DSM."""
    X = X.reshape(X.shape[0], -1)
    dsm_X = distance.pdist(X, metric=data_dsm_metric)

    if rsa_metric == 'spearman':
        r, _ = stats.spearmanr(dsm_X.ravel(), dsm_Y.ravel())
    else:
        r, _ = stats.pearsonr(dsm_X.ravel(), dsm_Y.ravel())

    return r


def compute_dsm(X, metric='correlation'):
    """Compute a dissimilarity matrix (DSM).

    Parameters
    ----------
    X : ndarray, shape (n_items, ...)
        For each item, all the features. Items are along the first dimension,
        all other dimensions will be flattened and treated as features.
    metric : str
        The metric to use to compute the DSM. Can be any metric supported by
        :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'.

    Returns
    -------
    dsm : ndarray, shape (n_items * n_items-1,)
        The DSM, in condensed form.
        See :func:`scipy.spatial.distance.squareform`.
    """
    X = X.reshape(X.shape[0], -1)
    n_features = X.shape[1]

    # Be careful with certain metrics
    if metric in ['correlation', 'cosine'] and n_features == 1:
        raise ValueError("There is only a single feature, so "
                         "'correlation' or 'cosine' can not be used as "
                         "dsm_metric. Consider using 'euclidean' "
                         "instead.")
    return distance.pdist(X, metric=metric)
