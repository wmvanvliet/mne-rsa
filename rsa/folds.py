import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def _create_folds(X, y, n_folds=None):
    """Split the observations in X into stratified folds."""
    if y is None or n_folds == 1:
        # Making one fold is easy
        return X[None, ...]

    y_one_hot = _convert_to_one_hot(y)
    n_items = y_one_hot.shape[1]

    if n_folds is None:
        # Set n_folds to maximum value
        n_folds = len(X) // n_items

    folds = []
    for _, fold in StratifiedKFold(n_folds).split(X, y):
        folds.append(_compute_item_means(X[fold], y_one_hot[fold]))
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


def _compute_item_means(X, y_one_hot):
    """Compute the mean data for each item."""
    n_per_class = y_one_hot.sum(axis=0)
    sums = (X.T @ y_one_hot)
    return (sums / n_per_class).T
