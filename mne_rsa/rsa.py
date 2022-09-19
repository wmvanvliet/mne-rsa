# encoding: utf-8
"""
Methods to compute representational similarity analysis (RSA).
"""

import numpy as np
from scipy import stats
from joblib import Parallel, delayed

from .folds import create_folds
from .dsm import _ensure_condensed, compute_dsm, compute_dsm_cv
from .searchlight import searchlight

try:
    # Version 1.8.0 and up
    from scipy.stats._stats_py import _kendall_dis
except ImportError:
    from scipy.stats.stats import _kendall_dis


def _kendall_tau_a(x, y):
    """Compute Kendall's Tau metric, A-variant.

    Taken from scipy.stats.kendalltau and modified to be the tau-a variant.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same size,"
                         " found x-size %s and y-size %s" % (x.size, y.size))
    elif not x.size or not y.size:
        return np.nan  # Return NaN if arrays are empty

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
                (cnt * (cnt - 1.) * (cnt - 2)).sum(),
                (cnt * (cnt - 1.) * (2 * cnt + 5)).sum())

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype='intp')

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype='intp')

    dis = _kendall_dis(x, y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
    xtie, x0, x1 = count_rank_tie(x)     # ties in x, stats
    ytie, y0, y1 = count_rank_tie(y)     # ties in y, stats

    tot = (size * (size - 1)) // 2

    if xtie == tot or ytie == tot:
        return np.nan

    # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #               = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / tot
    # Limit range to fix computational errors
    tau = min(1., max(-1., tau))

    return tau


def _consolidate_masks(masks):
    if type(masks[0]) == slice:
        mask = slice(None)
    else:
        mask = masks[0]
        for m in masks[1:]:
            mask &= m
    return mask


def _partial_correlation(dsm_data, dsm_model, masks=None, type='pearson'):
    """Compute partial Pearson/Spearman correlation."""
    if len(dsm_model) == 1:
        raise ValueError('Need more than one model DSM to use partial '
                         'correlation as metric.')
    if type not in ['pearson', 'spearman']:
        raise ValueError("Correlation type must be either 'pearson' or "
                         "'spearman'")

    if masks is not None:
        mask = _consolidate_masks(masks)
        dsm_model = [dsm[mask] for dsm in dsm_model]
        dsm_data = dsm_data[mask]

    X = np.vstack([dsm_data] + dsm_model).T
    if type == 'spearman':
        X = np.apply_along_axis(stats.rankdata, 0, X)
    X = X - X.mean(axis=0)
    cov_X_inv = np.linalg.pinv(X.T @ X)
    norm = np.sqrt(np.outer(np.diag(cov_X_inv), np.diag(cov_X_inv)))
    R_partial = cov_X_inv / norm
    return -R_partial[0, 1:]


def rsa_gen(dsm_data_gen, dsm_model, metric='spearman', ignore_nan=False):
    """Generate RSA values between data and model DSMs.

    Will yield RSA scores for each data DSM.

    Parameters
    ----------
    dsm_data_gen : generator of ndarray, shape (n_items, n_items)
        The generator for data DSMs
    dsm_model : ndarray, shape (n_items, n_items) | list of ndarray
        The model DSM, or list of model DSMs.
    metric : str
        The RSA metric to use to compare the DSMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8

    Yields
    ------
    rsa_val : float | ndarray, shape (len(dsm_model),)
        For each data DSM, the representational similarity with the model DSM.
        When multiple model DSMs are specified, this will be a 1D array of
        similarities, comparing the data DSM with each model DSM.

    See also
    --------
    rsa
    """
    if type(dsm_model) == list:
        return_array = True
        dsm_model = [_ensure_condensed(dsm, 'dsm_model') for dsm in dsm_model]
    else:
        return_array = False
        dsm_model = [_ensure_condensed(dsm_model, 'dsm_model')]

    if ignore_nan:
        masks = [~np.isnan(dsm) for dsm in dsm_model]
    else:
        masks = [slice(None)] * len(dsm_model)

    for dsm_data in dsm_data_gen:
        dsm_data = _ensure_condensed(dsm_data, 'dsm_data')
        if ignore_nan:
            data_mask = ~np.isnan(dsm_data)
            masks = [m & data_mask for m in masks]
        rsa_vals = _rsa_single_dsm(dsm_data, dsm_model, metric, masks)
        if return_array:
            yield np.asarray(rsa_vals)
        else:
            yield rsa_vals[0]


def _rsa_single_dsm(dsm_data, dsm_model, metric, masks):
    """Compute RSA between a single data DSM and one or more model DSMs."""
    if metric == 'spearman':
        rsa_vals = [stats.spearmanr(dsm_data[mask], dsm_model_[mask])[0]
                    for dsm_model_, mask in zip(dsm_model, masks)]
    elif metric == 'pearson':
        rsa_vals = [stats.pearsonr(dsm_data[mask], dsm_model_[mask])[0]
                    for dsm_model_, mask in zip(dsm_model, masks)]
    elif metric == 'kendall-tau-a':
        rsa_vals = [_kendall_tau_a(dsm_data[mask], dsm_model_[mask])
                    for dsm_model_, mask in zip(dsm_model, masks)]
    elif metric == 'partial':
        rsa_vals = _partial_correlation(dsm_data, dsm_model, masks)
    elif metric == 'partial-spearman':
        rsa_vals = _partial_correlation(dsm_data, dsm_model, masks,
                                        type='spearman')
    elif metric == 'regression':
        mask = _consolidate_masks(masks)
        dsm_model = [dsm[mask] for dsm in dsm_model]
        dsm_data = dsm_data[mask]
        X = np.atleast_2d(np.array(dsm_model)).T
        X = X - X.mean(axis=0)
        y = dsm_data - dsm_data.mean()
        rsa_vals = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        raise ValueError("Invalid RSA metric, must be one of: 'spearman', "
                         "'pearson', 'partial', 'partial-spearman', "
                         "'regression' or 'kendall-tau-a'.")
    return rsa_vals


def rsa(dsm_data, dsm_model, metric='spearman', ignore_nan=False,
        n_data_dsms=None, n_jobs=1, verbose=False):
    """Perform RSA between data and model DSMs.

    Parameters
    ----------
    dsm_data : ndarray, shape (n_items, n_items) | list | generator
        The data DSM (or list/generator of data DSMs).
    dsm_model : ndarray, shape (n_items, n_items) | list of ndarray
        The model DSM (or list of model DSMs).
    metric : str
        The RSA metric to use to compare the DSMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    n_data_dsms : int | None
        The number of data DSMs. This is useful when displaying a progress bar,
        so an estimate can be made of the computation time remaining. This
        information is available if ``dsm_data`` is an array or a list, but if
        it is a generator, this information is not available and you may want
        to set it explicitly.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa_val : float | ndarray, shape (len(dsm_data), len(dsm_model))
        Depending on whether one or more data and model DSMs were specified, a
        single similarity value or a 2D array of similarity values for each
        data DSM versus each model DSM.

    See also
    --------
    rsa_gen
    """
    return_array = False
    if type(dsm_data) == list or hasattr(dsm_data, '__next__'):
        return_array = True
    else:
        dsm_data = [dsm_data]

    if verbose:
        from tqdm import tqdm
        if n_data_dsms is not None:
            total = n_data_dsms
        elif hasattr(dsm_data, '__len__'):
            total = len(dsm_data)
        else:
            total = None
        dsm_data = tqdm(dsm_data, total=total, unit='DSM')

    if n_jobs == 1:
        rsa_vals = list(rsa_gen(dsm_data, dsm_model, metric, ignore_nan))
    else:
        def process_single_dsm(dsm):
            return next(rsa_gen([dsm], dsm_model, metric, ignore_nan))
        rsa_vals = Parallel(n_jobs)(delayed(process_single_dsm)(dsm)
                                    for dsm in dsm_data)
    if return_array:
        return np.asarray(rsa_vals)
    else:
        return rsa_vals[0]


def rsa_array(X, dsm_model, patches=None, data_dsm_metric='correlation',
              data_dsm_params=dict(), rsa_metric='spearman', ignore_nan=False,
              y=None, n_folds=1, n_jobs=1, verbose=False):
    """Perform RSA on an array of data, possibly in a searchlight pattern.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, n_times)
        An array containing the data.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`compute_dsm` to compute DSMs.
    patches : generator of tuples | None
        Searchlight patches as generated by :class:`searchlight`. If ``None``,
        no searchlight is used. Defaults to ``None``.
    data_dsm_metric : str
        The metric to use to compute the data DSMs. This can be any metric
        supported by the scipy.distance.pdist function. Defaults to
        'correlation'.
    data_dsm_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    rsa_metric : str
        The RSA metric to use to compare the DSMs. Valid options are:

        * 'spearman' for Spearman's correlation (the default)
        * 'pearson' for Pearson's correlation
        * 'kendall-tau-a' for Kendall's Tau (alpha variant)
        * 'partial' for partial Pearson correlations
        * 'partial-spearman' for partial Spearman correlations
        * 'regression' for linear regression weights

        Defaults to 'spearman'.
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    y : ndarray of int, shape (n_items,) | None
        For each item, a number indicating the class to which the item belongs.
        When ``None``, each item is assumed to belong to a different class.
        Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa_vals : ndarray, shape ([n_series,] [n_times,] [n_model_dsms])
        The RSA value for each searchlight patch. When ``spatial_radius`` is
        set to ``None``, there will only be no ``n_series`` dimension. When
        ``temporal_radius`` is set to ``None``, there will be no time
        dimension. When multiple models have been supplied, the last dimension
        will contain RSA results for each model.

    See Also
    --------
    searchlight
    compute_dsm
    dsm_array
    """
    if patches is None:
        patches = searchlight(X.shape)  # One big searchlight patch

    # Create folds for cross-validated DSM metrics
    X = create_folds(X, y, n_folds)
    # The data is now folds x items x n_series x n_times

    if type(dsm_model) == list:
        dsm_model = [_ensure_condensed(dsm, 'dsm_model') for dsm in dsm_model]
    else:
        dsm_model = [_ensure_condensed(dsm_model, 'dsm_model')]

    if ignore_nan:
        masks = [~np.isnan(dsm) for dsm in dsm_model]
    else:
        masks = [slice(None)] * len(dsm_model)

    if verbose:
        from tqdm import tqdm
        shape = getattr(patches, 'shape', (-1,))
        patches = tqdm(patches, unit='patch')
        try:
            setattr(patches, 'shape', shape)
        except AttributeError:
            pass

    def rsa_single_patch(patch):
        """Compute RSA for a single searchlight patch."""
        if len(X) == 1:  # Check number of folds
            # No cross-validation
            dsm_data = compute_dsm(X[0][patch],
                                   data_dsm_metric, **data_dsm_params)
        else:
            # Use cross-validation
            dsm_data = compute_dsm_cv(X[(slice(None),) + patch],
                                      data_dsm_metric, **data_dsm_params)
        if ignore_nan:
            data_mask = ~np.isnan(dsm_data)
            patch_masks = [m & data_mask for m in masks]
        else:
            patch_masks = masks
        return _rsa_single_dsm(dsm_data, dsm_model, rsa_metric, patch_masks)

    # Call RSA multiple times in parallel for each searchlight patch
    data = Parallel(n_jobs)(delayed(rsa_single_patch)(patch)
                            for patch in patches)

    # Figure out the desired dimensions of the resulting array
    dims = getattr(patches, 'shape', (-1,))
    if len(dsm_model) > 1:
        dims = dims + (len(dsm_model),)

    return np.array(data).reshape(dims)
