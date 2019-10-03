import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed


def _create_folds(X, y, n_folds=None, n_jobs=1):
    """Split the observations in X into stratified folds."""
    if y is None:
        # No folding
        return X[np.newaxis, ...]

    y_one_hot = _convert_to_one_hot(y)
    n_items = y_one_hot.shape[1]

    if n_folds is None:
        # Set n_folds to maximum value
        n_folds = len(X) // n_items

    if n_folds == 1:
        # Making one fold is easy
        folds = [_compute_item_means(X, y_one_hot)]
    else:
        # Computing the mean for each fold can be slow on big datasets. Hence
        # we allow for multiple cores to be used.
        folds = Parallel(n_jobs)(
            delayed(_compute_item_means)(X, y_one_hot, fold)
            for _, fold in StratifiedKFold(n_folds).split(X, y)
        )
    return np.array(folds)


def _convert_to_one_hot(y):
    """Convert y to a one-hot version if necessary."""
    if y.ndim == 1:
        y = y[:, np.newaxis]

    if y.ndim == 2 and y.shape[1] == 1:
        # y needs to be converted
        enc = OneHotEncoder(categories='auto').fit(y)
        return enc.transform(y).toarray()
    elif y.ndim > 2:
        raise ValueError('Wrong number of dimensions for `y`.')
    else:
        # y is probably already in one-hot form. We're not going to test this
        # explicitly, as it would take too long.
        return y


def _compute_item_means(X, y_one_hot, fold=slice(None, None)):
    """Compute the mean data for each item."""
    X = X[fold]
    y_one_hot = y_one_hot[fold]
    n_per_class = y_one_hot.sum(axis=0)
    sums = (X.T @ y_one_hot)
    return (sums / n_per_class).T
