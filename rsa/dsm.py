import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA


def compute_dsm(data, pca=False, metric='sqeuclidean', **kwargs):
    """Compute a dissimilarity matrix (DSM).

    Parameters
    ----------
    data : ndarray, shape (n_items, ...)
        For each item, all the features. The first are the items and all other
        dimensions will be flattened and treated as features.
    pca : bool
        Whether to use PCA when n_features > n_items. Defaults to False.
    metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'sqeuclidean'.
    **kwargs : dict, optional
        Extra arguments for the distance metric. The `cv_sqeuclidean` metric
        takes a parameter `cv` to indicate the number of folds to use during
        the cross-validation. Refer to :mod:`scipy.spatial.distance` for a list
        of all other metrics and their arguments.

    Returns
    -------
    dsm : ndarray, shape (n_classes * n_classes-1,)
        The cross-validated DSM, in condensed form.
        See :func:`scipy.spatial.distance.squareform`.
    """
    X = np.reshape(data, (len(data), -1))
    n_items, n_features = X.shape

    if pca and n_features > n_items:
        X = PCA(n_items).fit_transform(X)

    # Be careful with certain metrics
    if n_features == 1 and metric in ['correlation', 'cosine']:
        raise ValueError("There is only a single feature, so "
                         "'correlataion' and 'cosine' can not be "
                         "used as DSM metric. Consider using 'sqeuclidean' "
                         "instead.")

    return distance.pdist(X, metric)


def compute_dsm_cv(folds, metric='sqeuclidean', **kwargs):
    """Compute a dissimilarity matrix (DSM) using cross-validation.

    Parameters
    ----------
    folds : ndarray, shape (n_folds, n_items, ...)
        For each item, all the features. The first dimension are the folds used
        for cross-validation, items are along the second dimension, and all
        other dimensions will be flattened and treated as features.
    metric : str
        The metric to use to compute the data DSM. Can be any metric supported
        by :func:`scipy.spatial.distance.pdist`. Defaults to 'sqeuclidean'.
    **kwargs : dict, optional
        Extra arguments for the distance metric. The `cv_sqeuclidean` metric
        takes a parameter `cv` to indicate the number of folds to use during
        the cross-validation. Refer to :mod:`scipy.spatial.distance` for a list
        of all other metrics and their arguments.

    Returns
    -------
    dsm : ndarray, shape (n_classes * n_classes-1,)
        The cross-validated DSM, in condensed form.
        See :func:`scipy.spatial.distance.squareform`.
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
    #fold_selection = np.ones((n_folds,), dtype=np.bool)
    for test_fold in range(n_folds):
        X_test = X[test_fold]
        X_train = X_mean - (X_mean - X_test) / (n_folds - 1)
        #fold_selection[test_fold] = False
        #X_train = np.mean(X[fold_selection], axis=0)
        #fold_selection[test_fold] = True

        dist = distance.cdist(X_train, X_test, metric)
        dsm += dist[np.triu_indices_from(dist, 1)]

    return dsm / n_folds


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
