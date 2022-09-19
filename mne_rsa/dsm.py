# encoding: utf-8
"""
Methods to compute dissimilarity matrices (DSMs).
"""

import numpy as np
from scipy.spatial import distance

from .folds import create_folds
from .searchlight import searchlight


def compute_dsm(data, metric='correlation', **kwargs):
    """Compute a dissimilarity matrix (DSM).

    Parameters
    ----------
    data : ndarray, shape (n_items, ...)
        For each item, all the features. The first dimension are the items and
        all other dimensions will be flattened and treated as features.
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
                         "'correlation' and 'cosine' can not be "
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

    if not isinstance(dsm, np.ndarray):
        raise TypeError('A single DSM should be a NumPy array. '
                        'Multiple DSMs should be a list of NumPy arrays.')

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


class dsm_array:
    """Generate DSMs from an array of data, possibly in a searchlight pattern.

    First use :class:`searchlight` to compute the searchlight patches.
    Then you can use this function to compute DSMs for each searchlight patch.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, n_times)
        An array containing the data.
    patches : generator of tuples | None
        Searchlight patches as generated by :class:`searchlight`. If ``None``,
        no searchlight is used. Defaults to ``None``.
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
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).

    Yields
    ------
    dsm : ndarray, shape (n_patches, n_items * (n_items - 1))
        A DSM (in condensed form) for each searchlight patch.

    Attributes
    ----------
    shape : tuple of int
        Multidimensional shape of the generted DSMs.

        This is useful for re-shaping the result obtained after consuming the
        this generator.

        For a spatio-temporal searchlight:
            Three elements: the number of time-series, number of time
            samples and length of a consensed DSM.
        For a spatial searchlight:
            Two element: the number of time-series and length of a condensed
            DSM.
        For a temporal searchlight:
            Two elements: the number of time-samples and length of a condensed
            DSM.
        For no searchlight:
            One element: the length of a condensed DSM.

    See also
    --------
    dsm
    rsa
    searchlight
    """
    def __init__(self, X, patches=None, dist_metric='correlation',
                 dist_params=dict(), y=None, n_folds=1):
        if patches is None:
            patches = searchlight(X.shape)

        # Create folds for cross-validated DSM metrics
        self.X = create_folds(X, y, n_folds)
        # The data is now folds x items x n_series x ...

        self.patches = patches
        self.dist_metric = dist_metric
        self.dist_params = dist_params
        self.use_cv = len(self.X) > 1  # More than one fold present

        # Target shape for an array that would hold all of the generated DSMs.
        dsm_length = len(np.triu_indices(self.X.shape[1], k=1)[0])
        self.shape = patches.shape + (dsm_length,)

        # Setup the generator that will be producing the DSMs
        self._generator = self._iter_dsms()

    def __iter__(self):
        return self

    def __next__(self):
        """Generate a DSM for each searchlight patch."""
        return next(self._generator)

    def _iter_dsms(self):
        for patch in self.patches:
            if self.use_cv:
                yield compute_dsm_cv(self.X[(slice(None),) + patch],
                                     self.dist_metric, **self.dist_params)
            else:
                yield compute_dsm(self.X[0][patch],
                                  self.dist_metric, **self.dist_params)

    def __len__(self):
        """Get total number of DSMs that will be generated."""
        return len(self.patches)
