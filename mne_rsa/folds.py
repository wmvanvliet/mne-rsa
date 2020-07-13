from mne.utils import logger
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def _create_folds(X, y, n_folds=None):
    """Split the observations in X into stratified folds."""
    if y is None:
        # No folding
        return X[np.newaxis, ...]

    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError(f'The length of y ({len(y)}) does not match the '
                         f'number of items ({len(X)}).')

    y_one_hot = _convert_to_one_hot(y)
    n_items = y_one_hot.shape[1]

    if n_folds is None:
        # Set n_folds to maximum value
        n_folds = len(X) // n_items
        logger.info(f'Automatic dermination of folds: {n_folds}'
                    + ' (no cross-validation)' if n_folds == 1 else '')

    if n_folds == 1:
        # Making one fold is easy
        folds = [_compute_item_means(X, y_one_hot)]
    else:
        folds = []
        for _, fold in StratifiedKFold(n_folds).split(X, y):
            folds.append(_compute_item_means(X, y_one_hot, fold))
    return np.array(folds)


def _convert_to_one_hot(y):
    """Convert the labels in y to one-hot encoding."""
    y = np.asarray(y)
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


def _compute_item_means(X, y_one_hot, fold=slice(None)):
    """Compute the mean data for each item inside a fold."""
    X = X[fold]
    y_one_hot = y_one_hot[fold]
    n_per_class = y_one_hot.sum(axis=0)

    # The following computations go much faster when X is flattened.
    orig_shape = X.shape
    X_flat = X.reshape(len(X), -1)

    # Compute the mean for each item using matrix multiplication
    means = (y_one_hot.T @ X_flat) / n_per_class[:, np.newaxis]

    # Undo the flattening of X
    return means.reshape((len(means),) + orig_shape[1:])
