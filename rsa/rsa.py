# encoding: utf-8
"""
Module implementing private functions to perform RSA in searchlight patches.


Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
Ossi Lehtonen <ossi.lehtonen@aalto.fi>
"""

import numpy as np
from scipy import stats
from joblib import Parallel, delayed

from .dsm import compute_dsm, compute_dsm_cv, _ensure_condensed
from .folds import _create_folds


def rsa_spattemp(data, dsm_model, dist, spatial_radius, temporal_radius,
                 y=None, data_dsm_metric='sqeuclidean',
                 data_dsm_params=None, rsa_metric='spearman', n_folds=None,
                 n_jobs=1, verbose=False):
    """Compute spatio-temporal RSA using a searchlight pattern.

    Parameters
    ----------
    data : ndarray, shape (n_items, n_series, n_times)
        The brain data in the form of an ndarray. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`rsa.compute_dsm` to compute DSMs.
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
    n_folds : int | None
        Number of folds to use when using a cross-validation DSM metric.
        Defaults to ``None``, which means the maximum number of folds possible,
        given the data.
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

    See Also
    --------
    compute_dsm
    """
    n_series, n_samples = data.shape[1:]
    dsm_model = _ensure_condensed(dsm_model, 'dsm_model')

    # Create folds for cross-validated DSM metrics
    folds = _create_folds(data, y, n_folds, n_jobs)
    # The data is now folds x items x n_series x ...

    centers = _get_time_patch_centers(n_samples, temporal_radius)

    def patch_iterator():
        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=n_series * len(centers))
        for series in range(n_series):
            for sample in centers:
                patch = folds[
                    :, :,
                    np.flatnonzero(dist[series] < spatial_radius), ...,
                    sample - temporal_radius:sample + temporal_radius
                ]
                if verbose:
                    pbar.update(1)
                yield patch
        if verbose:
            pbar.close()

    # Run RSA for each spatial patch, use multiple cores.
    results = Parallel(n_jobs)(
        delayed(_rsa)(patch, dsm_model, data_dsm_metric, data_dsm_params,
                      rsa_metric)
        for patch in patch_iterator()
    )
    results = np.array(results)

    return results.reshape((n_series, len(centers)) + results.shape[1:])


def rsa_spat(data, dsm_model, dist, spatial_radius, y=None,
             data_dsm_metric='sqeuclidean', data_dsm_params=None,
             rsa_metric='spearman', n_folds=None, verbose=False):
    """Compute spatial RSA using a searchlight pattern.

    Computes RSA across space using a searchlight pattern, flattens the
    temporal dimension.

    Parameters
    ----------
    data : ndarray, shape (n_items, n_series, ...)
        The brain data in the form of an ndarray. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`rsa.compute_dsm` to compute DSMs.
    dist : ndarray, shape (n_series, n_series)
        The distances between all source points or sensors in meters.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters.
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs.
        When ``None``, each item is assumed to belong to a different class.
        Defaults to ``None``.
    data_dsm_metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'.
    data_dsm_params : dict | None
        Extra arguments for the distance metric used to compute the data DSM.
        The `cv_sqeuclidean` metric takes a parameter `cv` to indicate the
        number of folds to use during the cross-validation. Refer to
        :mod:`scipy.spatial.distance` for a list of all other metrics and their
        arguments. Defaults to ``None``.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs, either Spearman or
        Pearson correlation. Defaults to 'spearman'.
    n_folds : int | None
        Number of folds to use when using a cross-validation DSM metric.
        Defaults to ``None``, which means the maximum number of folds possible,
        given the data.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    results : ndarray, shape (n_series, n_times)
        The RSA correlation values for each spatio-temporal patch.

    See Also
    --------
    compute_dsm
    """
    dsm_model = _ensure_condensed(dsm_model, 'dsm_model')
    results = []

    # Create folds for cross-validated DSM metrics
    folds = _create_folds(data, y, n_folds)
    # The data is now folds x items x n_series x ...

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=len(dist))

    # Iterate over space
    for series_dist in dist:
        # Construct a searchlight patch of the given radius
        patch = folds[:, :, np.flatnonzero(series_dist < spatial_radius)]
        results.append(_rsa(patch, dsm_model, data_dsm_metric,
                            data_dsm_params, rsa_metric))
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    return np.array(results)


def rsa_temp(data, dsm_model, temporal_radius, y=None,
             data_dsm_metric='sqeuclidean', data_dsm_params=None,
             rsa_metric='spearman', n_folds=None, verbose=False):
    """Perform temporal RSA analysis using a searchlight in time.

    Computes RSA across time using a searchlight, flattens the spatial
    dimension.

    Parameters
    ---------
    data : ndarray, shape (n_items, ..., n_times)
        The brain data in the spatial patch. The last dimension should be
        consecutive time points.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`rsa.compute_dsm` to compute DSMs.
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs.
        When ``None``, each item is assumed to belong to a different class.
        Defaults to ``None``.
    data_dsm_metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'correlation'.
    data_dsm_params : dict | None
        Extra arguments for the distance metric used to compute the data DSM.
        The `cv_sqeuclidean` metric takes a parameter `cv` to indicate the
        number of folds to use during the cross-validation. Refer to
        :mod:`scipy.spatial.distance` for a list of all other metrics and their
        arguments. Defaults to ``None``.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs, either Spearman or
        Pearson correlation. Defaults to 'spearman'.
    n_folds : int | None
        Number of folds to use when using a cross-validation DSM metric.
        Defaults to ``None``, which means the maximum number of folds possible,
        given the data.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    results : ndarray, shape (n_temporal_patches,)
        The RSA correlation values for each temporal patch.

    See Also
    --------
    compute_dsm
    """
    n_samples = data.shape[-1]
    dsm_model = _ensure_condensed(dsm_model, 'dsm_model')

    # Iterate over time
    results = []
    centers = _get_time_patch_centers(n_samples, temporal_radius)

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=len(centers))

    # Create folds for cross-validated DSM metrics
    folds = _create_folds(data, y, n_folds)
    # The data is now folds x items x ... x n_samples

    for center in centers:
        # Construct a searchlight patch of the given radius
        patch = folds[..., center - temporal_radius:center + temporal_radius]
        results.append(_rsa(patch, dsm_model, data_dsm_metric,
                            data_dsm_params, rsa_metric))
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


def _kendall_tau_a(a, b):
    n = len(a)
    K = 0
    for k in range(0, n - 1):
        pair_relations_a = np.sign(a[k] - a[k + 1:])
        pair_relations_b = np.sign(b[k] - b[k + 1:])
        K += np.sum(pair_relations_a * pair_relations_b)
    return K / (n * (n - 1) / 2)


def _rsa(folds, dsm_model, data_dsm_metric='sqeuclidean',
         data_dsm_params=None, rsa_metric='spearman'):
    """Perform RSA between some data and a model DSM."""
    if not data_dsm_params:
        data_dsm_params = dict()

    if len(folds) == 1:
        dsm_data = compute_dsm(folds[0], metric=data_dsm_metric,
                               **data_dsm_params)
    else:
        dsm_data = compute_dsm_cv(folds, metric=data_dsm_metric,
                                  **data_dsm_params)

    if type(dsm_model) is not list:
        dsm_model = [_ensure_condensed(dsm_model)]
    else:
        dsm_model = [_ensure_condensed(dsm) for dsm in dsm_model]

    if rsa_metric == 'spearman':
        rs = [stats.spearmanr(dsm_data, dsm_model_)[0]
              for dsm_model_ in dsm_model]
    elif rsa_metric == 'pearson':
        rs = [stats.pearsonr(dsm_data, dsm_model_)[0]
              for dsm_model_ in dsm_model]
    elif rsa_metric == 'kendall-tau-a':
        rs = [_kendall_tau_a(dsm_data, dsm_model_)
              for dsm_model_ in dsm_model]
    elif rsa_metric == 'partial':
        if len(dsm_model) == 1:
            raise ValueError('Need more than one model DSM to use partial '
                             'correlation as metric.')
        X = np.hstack([dsm_data] + dsm_model).T
        X -= X.mean(axis=0)
        cov_X_inv = np.linalg.pinv(X.T @ X)
        cov_X_inv_diag = np.diag(cov_X_inv)
        R_partial = cov_X_inv / np.sqrt(np.outer(cov_X_inv_diag))
        rs = -R_partial[0, 1:]
    elif rsa_metric == 'regression':
        X = np.atleast_2d(np.hstack(dsm_model)).T
        X -= X.mean(axis=0)
        rs = np.linalg.lstsq(X, dsm_data, rcond=None)[0]
    else:
        raise ValueError("Invalid rsa_metric, must be one of: 'spearman', "
                         "'pearson', 'partial', 'regression' or "
                         "'kendall-tau-a'.")
    return rs
