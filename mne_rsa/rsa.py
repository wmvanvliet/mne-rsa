# encoding: utf-8
"""
Methods to compute representational similarity analysis (RSA).
"""

from types import GeneratorType
import numpy as np
from scipy import stats
from scipy.stats.stats import _kendall_dis
from joblib import Parallel, delayed

from .dsm import _ensure_condensed, dsm_array


def _kendall_tau_a(x, y):
    """Compute Kendall's Tau metric, A-variant.

    Taken from scipy.stats.kendalltau and modified to be the tau-a variant.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("All inputs to `kendalltau` must be of the same size, "
                         "found x-size %s and y-size %s" % (x.size, y.size))
    elif not x.size or not y.size:
        return np.nan  # Return NaN if arrays are empty

    def count_rank_tie(ranks):
        cnt = np.bincount(ranks).astype('int64', copy=False)
        cnt = cnt[cnt > 1]
        return ((cnt * (cnt - 1) // 2).sum(),
                (cnt * (cnt - 1.) * (cnt - 2)).sum(),
                (cnt * (cnt - 1.) * (2*cnt + 5)).sum())

    size = x.size
    perm = np.argsort(y)  # sort on y and convert y to dense ranks
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    # stable sort on x and convert x to dense ranks
    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

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


def _partial_correlation(dsm_data, dsm_model, type='pearson'):
    """Compute partial Pearson/Spearman correlation."""
    if len(dsm_model) == 1:
        raise ValueError('Need more than one model DSM to use partial '
                         'correlation as metric.')
    if type not in ['pearson', 'spearman']:
        raise ValueError("Correlation type must by either 'pearson' or "
                         "'spearman'")
    X = np.vstack([dsm_data] + dsm_model).T
    if type == 'spearman':
        X = np.apply_along_axis(stats.rankdata, 0, X)
    X = X - X.mean(axis=0)
    cov_X_inv = np.linalg.pinv(X.T @ X)
    norm = np.sqrt(np.outer(np.diag(cov_X_inv), np.diag(cov_X_inv)))
    R_partial = cov_X_inv / norm
    return -R_partial[0, 1:]


def rsa_gen(dsm_data_gen, dsm_model, metric='spearman'):
    """Generate RSA values between data and model DSMs.

    Will yield RSA scores for each data DSM.

    Parameters
    ----------
    dsm_data_gen : generator of ndarray, shape (n_items, n_items)
        The generator function for data DSMs
    dsm_model : ndarray, shape (n_items, n_items) | list of ndarray
        The model DSM, or list of model DSMs.
    metric : str
        The RSA metric to use to compare the DSMs. Valid options are::
         - 'spearman' for Spearman's correlation (the default)
         - 'pearson' for Pearson's correlation
         - 'kendall-tau-a' for Kendall's Tau (alpha variant)
         - 'partial' for partial Pearson correlations
         - 'partial-spearman' for partial Spearman correlations
         - 'regression' for linear regression weights
        Defaults to 'spearman'

    Yields
    -------
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

    for dsm_data in dsm_data_gen:
        dsm_data = _ensure_condensed(dsm_data, 'dsm_data')
        if metric == 'spearman':
            rsa_vals = [stats.spearmanr(dsm_data, dsm_model_)[0]
                        for dsm_model_ in dsm_model]
        elif metric == 'pearson':
            rsa_vals = [stats.pearsonr(dsm_data, dsm_model_)[0]
                        for dsm_model_ in dsm_model]
        elif metric == 'kendall-tau-a':
            rsa_vals = [_kendall_tau_a(dsm_data, dsm_model_)
                        for dsm_model_ in dsm_model]
        elif metric == 'partial':
            rsa_vals = _partial_correlation(dsm_data, dsm_model)
        elif metric == 'partial-spearman':
            rsa_vals = _partial_correlation(dsm_data, dsm_model,
                                            type='spearman')
        elif metric == 'regression':
            X = np.atleast_2d(np.array(dsm_model)).T
            X = X - X.mean(axis=0)
            rsa_vals = np.linalg.lstsq(X, dsm_data, rcond=None)[0]
        else:
            raise ValueError("Invalid RSA metric, must be one of: 'spearman', "
                             "'pearson', 'partial', 'regression' or "
                             "'kendall-tau-a'.")
        if return_array:
            yield np.asarray(rsa_vals)
        else:
            yield rsa_vals[0]


def rsa(dsm_data, dsm_model, metric='spearman'):
    """Perform RSA between data and model DSMs.

    Parameters
    ----------
    dsm_data : ndarray, shape (n_items, n_items) | list | generator
        The data DSM (or list/generator of data DSMs).
    dsm_model : ndarray, shape (n_items, n_items) | list of ndarray
        The model DSM (or list of model DSMs).
    metric : str
        The RSA metric to use to compare the DSMs. Valid options are::
         - 'spearman' for Spearman's correlation
         - 'pearson' for Pearson's correlation
         - 'kendall-tau-a' for Kendall's Tau (alpha variant)
         - 'partial' for partial Pearson correlations
         - 'regression' for linear regression weights

    Returns
    -------
    rsa_val : float | ndarray, shape (len(dsm_data), len(dsm_model))
        Depending on whether one or mode data and model DSMs were specified, a
        single similarity value or a 2D array of similarity values for each
        data DSM versus each model DSM.

    See also
    --------
    rsa_gen
    """
    return_array = False
    if type(dsm_data) == list or isinstance(dsm_data, GeneratorType):
        return_array = True
    else:
        dsm_data = [dsm_data]

    rsa_vals = list(rsa_gen(dsm_data, dsm_model, metric))

    if return_array:
        return np.asarray(rsa_vals)
    else:
        return rsa_vals[0]


def rsa_array(X, dsm_model, dist=None, spatial_radius=None,
              temporal_radius=None, data_dsm_metric='correlation',
              data_dsm_params=dict(), rsa_metric='spearman', y=None,
              n_folds=1, sel_series=None, sel_times=None, n_jobs=1,
              verbose=False):
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

        Use :func:`rsa.compute_dsm` to compute DSMs.
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
    data_dsm_metric : str
        The metric to use to compute the data DSMs. This can be any metric
        supported by the scipy.distance.pdist function. Defaults to
        'correlation'.
    data_dsm_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs. This can either be
        'spearman' correlation or 'pearson' correlation.
        Defaults to 'spearman'.
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
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa_vals : ndarray, shape (n_series, n_times[, n_model_dsms])
        The RSA value for each searchlight patch. When ``spatial_radius`` is
        set to ``None``, there will only be one series. When
        ``temporal_radius`` is set to ``None``, there will only be one time
        point. When multiple models have been supplied, the last dimension will
        contain RSA results for each model.

    See Also
    --------
    compute_dsm
    """
    # Joblib does not support passing a generator as a function argument.
    # To work around this, we wrap the call to rsa() inside a temporary
    # function.
    def call_rsa(sel_series, sel_times, position):
        return rsa(
            dsm_data=dsm_array(
                X, dist, spatial_radius, temporal_radius, data_dsm_metric,
                data_dsm_params, y, n_folds, sel_series=sel_series,
                sel_times=sel_times, verbose=position),
            dsm_model=dsm_model,
            metric=rsa_metric)

    # Deal with subselection of data
    if sel_series is None:
        sel_series = np.arange(X.shape[1])
    if sel_times is None:
        sel_times = np.arange(X.shape[-1])
    n_series = len(sel_series)

    # Call RSA multiple times in parallel. Each thread computes the RSA on part
    # of the data.
    if spatial_radius is not None and n_series >= n_jobs:
        # Split the data along series
        series_chunks = _split(sel_series, n_jobs)
        data = Parallel(n_jobs, verbose=1 if verbose else 0)(
            delayed(call_rsa)(chunk, None, i)
            for i, chunk in enumerate(series_chunks, 1))
    elif temporal_radius is not None:
        # Split the data along time points
        times_chunks = _split(sel_times, n_jobs)
        data = Parallel(n_jobs, verbose=1 if verbose else 0)(
            delayed(call_rsa)(None, chunk, i)
            for i, chunk in enumerate(times_chunks, 1))
    else:
        # No searchlight. No parallel processing.
        data = call_rsa(None, None, 1)

    # Collect the RSA values that were computed in the different threads into
    # one array.
    if spatial_radius is not None and temporal_radius is not None:
        data = np.vstack(data)
        data = data.reshape((n_series, -1) + data.shape[1:])
    elif spatial_radius is not None:
        data = np.vstack(data)[:, np.newaxis, ...]
    elif temporal_radius is not None:
        data = np.vstack(data)[np.newaxis, ...]
    else:
        data = data[np.newaxis, np.newaxis, ...]

    return data


def _split(x, n):
    """Split x into n chunks. The last chunk may contain less items."""
    chunk_size = int(np.ceil(len(x) / n))
    return [x[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
