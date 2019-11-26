# encoding: utf-8
"""
Methods to compute representational similarity analysis (RSA).
"""

from types import GeneratorType
import numpy as np
from scipy import stats
from joblib import Parallel, delayed

from .dsm import (_ensure_condensed, dsm_spattemp, dsm_spat, dsm_temp,
                  compute_dsm, compute_dsm_cv)
from .folds import _create_folds


def _kendall_tau_a(a, b):
    """Compute Kendall's Tau metric, alpha variant."""
    n = len(a)
    K = 0
    for k in range(0, n - 1):
        pair_relations_a = np.sign(a[k] - a[k + 1:])
        pair_relations_b = np.sign(b[k] - b[k + 1:])
        K += np.sum(pair_relations_a * pair_relations_b)
    return K / (n * (n - 1) / 2)


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
         - 'spearman' for Spearman's correlation
         - 'pearson' for Pearson's correlation
         - 'kendall-tau-a' for Kendall's Tau (alpha variant)
         - 'partial' for partial Pearson correlations
         - 'regression' for linear regression weights

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
            if len(dsm_model) == 1:
                raise ValueError('Need more than one model DSM to use partial '
                                 'correlation as metric.')
            X = np.vstack([dsm_data] + dsm_model).T
            X = X - X.mean(axis=0)
            cov_X_inv = np.linalg.pinv(X.T @ X)
            norm = np.sqrt(np.outer(np.diag(cov_X_inv), np.diag(cov_X_inv)))
            R_partial = cov_X_inv / norm
            rsa_vals = -R_partial[0, 1:]
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
              data_dsm_params=None, rsa_metric='spearman', y=None,
              n_folds=None, n_jobs=1, verbose=False):
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
    stc_dsm_metric : str
        The metric to use to compute the data DSMs. This can be any metric
        supported by the scipy.distance.pdist function. Defaults to
        'correlation'.
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
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    stc : SourceEstimate | list of SourceEstimate
        The correlation values for each searchlight patch. When spatial_radius
        is set to None, there will only be one vertex. When temporal_radius is
        set to None, there will only be one time point. When multiple models
        have been supplied, a list will be returned containing the RSA results
        for each model.

    See Also
    --------
    compute_dsm
    """
    # Spatio-Temporal RSA
    if spatial_radius is not None and temporal_radius is not None:
        if dist is None:
            raise ValueError('A spatial radius was requested, but no distance '
                             'information was specified (=dist parameter).')
        n_series = X.shape[1]
        # Split the data into chunks. Each chunk will be processed in parallel.
        chunks = _split(np.arange(n_series), n_jobs)

        # Joblib does not support passing a generator as a function argument.
        # To work around this, we wrap the call to rsa() inside a temporary
        # function.
        def call_rsa(sel_series, position):
            return rsa(
                dsm_data=dsm_spattemp(
                    X, dist, spatial_radius, temporal_radius, data_dsm_metric,
                    data_dsm_params, y, n_folds, sel_series, verbose=position),
                dsm_model=dsm_model,
                metric=rsa_metric)

        data = Parallel(n_jobs, verbose=1 if verbose else 0)(
            delayed(call_rsa)(sel_series, i)
            for i, sel_series in enumerate(chunks, 1))

        data = np.vstack(data)
        data = data.reshape((n_series, -1) + data.shape[1:])

    # Spatial RSA
    elif spatial_radius is not None:
        if dist is None:
            raise ValueError('A spatial radius was requested, but no distance '
                             'information was specified (=dist parameter).')
        # Split the data into chunks. Each chunk will be processed in parallel.
        chunks = _split(np.arange(len(X)), n_jobs)

        # Joblib does not support passing a generator as a function argument.
        # To work around this, we wrap the call to rsa() inside a temporary
        # function.
        def call_rsa(sel_series, position):
            return rsa(
                dsm_data=dsm_spat(
                    X, dist, spatial_radius, data_dsm_metric, data_dsm_params,
                    y, n_folds, sel_series, verbose=position),
                dsm_model=dsm_model,
                metric=rsa_metric)

        data = Parallel(n_jobs, verbose=1 if verbose else 0)(
            delayed(call_rsa)(sel_series, i)
            for i, sel_series in enumerate(chunks, 1))
        data = np.vstack(data)[:, np.newaxis, ...]

    # Temporal RSA
    elif temporal_radius is not None:
        # Split the data into chunks. Each chunk will be processed in parallel.
        chunks = _split(np.arange(X.shape[-1]), n_jobs)

        # Joblib does not support passing a generator as a function argument.
        # To work around this, we wrap the call to rsa() inside a temporary
        # function.
        def call_rsa(sel_series, position):
            return rsa(
                dsm_data=dsm_temp(
                    X, dist, spatial_radius, data_dsm_metric, data_dsm_params,
                    y, n_folds, sel_series, verbose=position),
                dsm_model=dsm_model,
                metric=rsa_metric)

        data = Parallel(n_jobs, verbose=1 if verbose else 0)(
            delayed(call_rsa)(sel_times, i)
            for i, sel_times in enumerate(chunks, 1))
        data = np.vstack(data)[np.newaxis, ...]

    # RSA between two DSMs
    else:
        folds = _create_folds(X, y, n_folds)
        if len(folds) == 1:
            dsm_func = compute_dsm
        else:
            dsm_func = compute_dsm_cv
        data = rsa(
            dsm_data=dsm_func(X, data_dsm_metric, data_dsm_params),
            dsm_model=dsm_model,
            metric=rsa_metric)
        data = data[np.newaxis, np.newaxis, ...]

    return data


def _split(x, n):
    """Split x into n chunks. The last chunk may contain less items."""
    chunk_size = int(np.ceil(len(x) / n))
    return [x[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
