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
    X = np.reshape(np.asarray(data), (len(data), -1))
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
            raise ValueError(f'Invalid dimensions for "{var_name}" '
                             '({dsm.shape}). The DSM should either be a '
                             'square matrix, or a one dimensional array when '
                             'in condensed form.')
        dsm = distance.squareform(dsm)
    elif dsm.ndim != 1:
        raise ValueError(f'Invalid dimensions for "{var_name}" ({dsm.shape}). '
                         'The DSM should either be a square matrix, or a one '
                         'dimensional array when in condensed form.')
    return dsm


def _n_items_from_dsm(dsm):
    """Get the number of items, given a DSM."""
    if dsm.ndim == 2:
        return dsm.shape[0]
    elif dsm.ndim == 1:
        return distance.squareform(dsm).shape[0]


def dsm_searchlight(X, patches, dist_metric='correlation', dist_params=dict(),
                    y=None, n_folds=1, sel_series=None, sel_samples=None):
    """Generate DSMs from an array of data in a searchlight pattern.

    First use :func:`searchlight_patches` to compute the searchlight patches.
    Then you can use this function to compute DSMs for each searchlight patch.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, n_times)
        An array containing the data.
    patches : generator of tuples
        Searchlight patches as generated by the :func:`searchlight_patches`
        function.
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

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each searchlight patch.

    See also
    --------
    searchlight_patches
    compute_dsm
    rsa_array
    """
    # Create folds for cross-validated DSM metrics
    X = _create_folds(X, y, n_folds)
    # The data is now folds x items x n_series x n_times

    # Compute a DSM for each searchlight patch
    for patch in patches:
        if n_folds == 1:
            dsm = compute_dsm(X[patch][0], dist_metric, **dist_params)
        else:
            dsm = compute_dsm_cv(X[patch], dist_metric, **dist_params)
        yield dsm
