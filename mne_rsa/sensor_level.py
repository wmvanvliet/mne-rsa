
"""
Module implementing representational similarity analysis (RSA) at the sensor
level.

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational
similarity analysis - connecting the branches of systems neuroscience.
Frontiers in Systems Neuroscience, 2(November), 4.
https://doi.org/10.3389/neuro.06.004.2008

Authors
-------
Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import numpy as np
from scipy.spatial import distance
import mne
from mne.utils import logger

from .dsm import _n_items_from_dsm, dsm_array
from .searchlight import searchlight
from .rsa import rsa_array


def rsa_evokeds(evokeds, dsm_model, noise_cov=None, spatial_radius=0.04,
                temporal_radius=0.1, evoked_dsm_metric='correlation',
                evoked_dsm_params=dict(), rsa_metric='spearman',
                ignore_nan=False, y=None, n_folds=1, picks=None, tmin=None,
                tmax=None, n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern on evokeds.

    The output is an Evoked object where the "signal" at each sensor is
    the RSA, computed using all surrounding sensors.

    Parameters
    ----------
    evokeds : list of :class:`mne.Evoked`
        The evoked brain activity for each item. If you have more than one
        Evoked object per item (i.e. repetitions), you can use the ``y``
        parameter to assign evokeds to items.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`compute_dsm` to compute DSMs.
    noise_cov : :class:`mne.Covariance` | None
        When specified, the data will by normalized using the noise covariance.
        This is recommended in all cases, but a hard requirement when the data
        contains sensors of different types. Defaults to None.
    spatial_radius : float | None
        The spatial radius of the searchlight patch in meters. All sensors
        within this radius will belong to the searchlight patch. Set to None to
        only perform the searchlight over time, flattening across sensors.
        Defaults to 0.04.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to 0.1.
    evoked_dsm_metric : str
        The metric to use to compute the DSM for the evokeds. This can be any
        metric supported by the scipy.distance.pdist function. See also the
        ``evoked_dsm_params`` parameter to specify and additional parameter for
        the distance function. Defaults to 'correlation'.
    evoked_dsm_params : dict
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
        For each Evoked, a number indicating the item to which it belongs.
        When ``None``, each Evoked is assumed to belong to a different item.
        Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings
        (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels. Can
        also be the string values "all" to pick all channels, or "data" to pick
        data channels. ``None`` (default) will pick all MEG and EEG channels,
        excluding those maked as "bad".
    tmin : float | None
        When set, searchlight patches will only be generated from subsequent
        time points starting from this time point. This value is given in
        seconds. Defaults to ``None``, in which case patches are generated
        starting from the first time point.
    tmax : float | None
        When set, searchlight patches will only be generated up to and
        including this time point. This value is given in seconds. Defaults to
        ``None``, in which case patches are generated up to and including the
        last time point.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa : Evoked | list of Evoked
        The correlation values for each searchlight patch. When spatial_radius
        is set to None, there will only be one virtual sensor. When
        temporal_radius is set to None, there will only be one time point. When
        multiple models have been supplied, a list will be returned containing
        the RSA results for each model.

    See Also
    --------
    compute_dsm
    """
    one_model = type(dsm_model) != list
    if one_model:
        dsm_model = [dsm_model]

    logger.info(f'Performing RSA between Evokeds and {len(dsm_model)} model '
                'DSM(s)')

    # Check for compatibility of the evokeds and the model features
    for dsm in dsm_model:
        n_items = _n_items_from_dsm(dsm)
        if len(evokeds) != n_items and y is None:
            raise ValueError(
                'The number of evokeds (%d) should be equal to the '
                'number of items in `dsm_model` (%d). Alternatively, use '
                'the `y` parameter to assign evokeds to items.'
                % (len(evokeds), n_items))
        if y is not None and np.unique(y) != n_items:
            raise ValueError(
                'The number of items in `dsm_model` (%d) does not match '
                'the number of items encoded in the `y` matrix (%d).'
                % (n_items, len(np.unique(y))))

    times = evokeds[0].times
    for evoked in evokeds:
        if np.any(evoked.times != times):
            raise ValueError('Not all evokeds have the same time points.')

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = round(evokeds[0].info['sfreq'] * temporal_radius)

    # Normalize with the noise cov
    if noise_cov is not None:
        logger.info('    Whitening data using noise covariance')
        diag = spatial_radius is not None
        evokeds = [mne.whiten_evoked(evoked, noise_cov, diag=diag)
                   for evoked in evokeds]

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in evokeds[0].info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    picks = mne.io.pick._picks_to_idx(evokeds[0].info, picks, none='data')
    if len(picks) != len(set(picks)):
        raise ValueError("`picks` are not unique. Please remove duplicates.")
    samples_from, samples_to = _tmin_tmax_to_indices(evokeds[0].times,
                                                     tmin, tmax)

    if spatial_radius is not None:
        logger.info(f'    Spatial radius: {spatial_radius} meters')
        logger.info(f'    Using {len(picks)} sensors')
    if temporal_radius is not None:
        logger.info(f'    Temporal radius: {temporal_radius} samples')
        logger.info(f'    Time inverval: {tmin}-{tmax} seconds')

    # Perform the RSA
    X = np.array([evoked.data for evoked in evokeds])
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius,
                          sel_series=picks, samples_from=samples_from,
                          samples_to=samples_to)
    data = rsa_array(X, dsm_model, patches, data_dsm_metric=evoked_dsm_metric,
                     data_dsm_params=evoked_dsm_params, rsa_metric=rsa_metric,
                     ignore_nan=ignore_nan, y=y, n_folds=n_folds,
                     n_jobs=n_jobs, verbose=verbose)

    # Pack the result in an Evoked object
    if spatial_radius is not None:
        info = mne.pick_info(evokeds[0].info, picks)
    else:
        info = mne.create_info(['rsa'], evokeds[0].info['sfreq'])
    tmin = _construct_tmin(evokeds[0].times, samples_from, samples_to,
                           temporal_radius)

    if one_model:
        return mne.EvokedArray(np.atleast_2d(data[..., 0]), info, tmin,
                               comment='RSA', nave=len(evokeds))
    else:
        return [mne.EvokedArray(np.atleast_2d(data[..., i]), info, tmin,
                                comment='RSA', nave=len(evokeds))
                for i in range(data.shape[-1])]


def rsa_epochs(epochs, dsm_model, noise_cov=None, spatial_radius=0.04,
               temporal_radius=0.1, epochs_dsm_metric='correlation',
               epochs_dsm_params=dict(), rsa_metric='spearman',
               ignore_nan=False, y=None, n_folds=1, picks=None, tmin=None,
               tmax=None, dropped_as_nan=False, n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern on epochs.

    The output is an Evoked object where the "signal" at each sensor is
    the RSA, computed using all surrounding sensors.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The brain activity during the epochs. The event codes are used to
        distinguish between items.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`compute_dsm` to compute DSMs.
    noise_cov : mne.Covariance | None
        When specified, the data will by normalized using the noise covariance.
        This is recommended in all cases, but a hard requirement when the data
        contains sensors of different types. Defaults to None.
    spatial_radius : floats | None
        The spatial radius of the searchlight patch in meters. All sensors
        within this radius will belong to the searchlight patch. Set to None to
        only perform the searchlight over time, flattening across sensors.
        Defaults to 0.04.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to 0.1.
    epochs_dsm_metric : str
        The metric to use to compute the DSM for the epochs. This can be any
        metric supported by the scipy.distance.pdist function. See also the
        ``epochs_dsm_params`` parameter to specify and additional parameter for
        the distance function. Defaults to 'correlation'.
    epochs_dsm_params : dict
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
        For each Epoch, a number indicating the item to which it belongs.
        When ``None``, the event codes are used to differentiate between items.
        Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings
        (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels. Can
        also be the string values "all" to pick all channels, or "data" to pick
        data channels. ``None`` (default) will pick all MEG and EEG channels,
        excluding those maked as "bad".
    tmin : float | None
        When set, searchlight patches will only be generated from subsequent
        time points starting from this time point. This value is given in
        seconds. Defaults to ``None``, in which case patches are generated
        starting from the first time point.
    tmax : float | None
        When set, searchlight patches will only be generated up to and
        including this time point. This value is given in seconds. Defaults to
        ``None``, in which case patches are generated up to and including the
        last time point.
    dropped_as_nan : bool
        When this is set to ``True``, the drop log will be used to inject NaN
        values in the DSMs at the locations where a bad epoch was dropped. This
        is useful to ensure the dimensions of the DSM are the same,
        irregardless of any bad epochs that were dropped. Make sure to use
        ``ignore_nan=True`` when using DSMs with NaNs in them during subsequent
        RSA computations. Defaults to ``False``.

        .. versionadded:: 0.8
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa : Evoked | list of Evoked
        The correlation values for each searchlight patch. When spatial_radius
        is set to None, there will only be one virtual sensor. When
        temporal_radius is set to None, there will only be one time point. When
        multiple models have been supplied, a list will be returned containing
        the RSA results for each model.

    See Also
    --------
    compute_dsm
    """
    one_model = type(dsm_model) is np.ndarray
    if one_model:
        dsm_model = [dsm_model]

    logger.info(f'Performing RSA between Epochs and {len(dsm_model)} model '
                'DSM(s)')

    if y is None:
        y_source = 'Epoch object'
        y = epochs.events[:, 2]
    else:
        y_source = '`y` matrix'

    # Check for compatibility of the epochs and the model features
    for dsm in dsm_model:
        n_items = _n_items_from_dsm(dsm)
        if len(np.unique(y)) != n_items:
            raise ValueError(
                'The number of items in `dsm_model` (%d) does not match '
                'the number of items encoded in the %s (%d).'
                % (n_items, y_source, len(np.unique(y))))

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = round(epochs.info['sfreq'] * temporal_radius)


    # # Normalize with the noise cov
    # if noise_cov is not None:
    #     diag = spatial_radius is not None
    #     evokeds = [mne.whiten_evoked(evoked, noise_cov, diag=diag)
    #                for evoked in evokeds]

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in epochs.info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    picks = mne.io.pick._picks_to_idx(epochs.info, picks, none='data')
    if len(picks) != len(set(picks)):
        raise ValueError("`picks` are not unique. Please remove duplicates.")
    samples_from, samples_to = _tmin_tmax_to_indices(epochs.times, tmin, tmax)

    if spatial_radius is not None:
        logger.info(f'    Spatial radius: {spatial_radius} meters')
        logger.info(f'    Using {len(picks)} sensors')
    if temporal_radius is not None:
        logger.info(f'    Temporal radius: {temporal_radius} samples')
        logger.info(f'    Time inverval: {tmin}-{tmax} seconds')

    # Perform the RSA
    X = epochs.get_data()
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius,
                          sel_series=picks, samples_from=samples_from,
                          samples_to=samples_to)
    data = rsa_array(X, dsm_model, patches, data_dsm_metric=epochs_dsm_metric,
                     data_dsm_params=epochs_dsm_params, rsa_metric=rsa_metric,
                     ignore_nan=ignore_nan, y=y, n_folds=n_folds,
                     n_jobs=n_jobs, verbose=verbose)

    # Pack the result in an Evoked object
    if spatial_radius is not None:
        info = epochs.info
        info = mne.pick_info(info, picks)
    else:
        info = mne.create_info(['rsa'], epochs.info['sfreq'])
    tmin = _construct_tmin(epochs.times, samples_from, samples_to,
                           temporal_radius)

    if one_model:
        return mne.EvokedArray(np.atleast_2d(data), info, tmin, comment='RSA',
                               nave=len(np.unique(y)))
    else:
        return [mne.EvokedArray(np.atleast_2d(data[..., i]), info, tmin,
                                comment='RSA', nave=len(np.unique(y)))
                for i in range(data.shape[-1])]


def dsm_evokeds(evokeds, noise_cov=None, spatial_radius=0.04,
                temporal_radius=0.1, dist_metric='correlation',
                dist_params=dict(), y=None, n_folds=1, picks=None, tmin=None,
                tmax=None):
    """Generate DSMs in a searchlight pattern on evokeds.

    Parameters
    ----------
    evokeds : list of mne.Evoked
        The evoked brain activity for each item. If you have more than one
        Evoked object per item (i.e. repetitions), you can use the ``y``
        parameter to assign evokeds to items.
    noise_cov : mne.Covariance | None
        When specified, the data will by normalized using the noise covariance.
        This is recommended in all cases, but a hard requirement when the data
        contains sensors of different types. Defaults to None.
    spatial_radius : floats | None
        The spatial radius of the searchlight patch in meters. All sensors
        within this radius will belong to the searchlight patch. Set to None to
        only perform the searchlight over time, flattening across sensors.
        Defaults to 0.04.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to 0.1.
    dist_metric : str
        The metric to use to compute the DSM for the evokeds. This can be any
        metric supported by the scipy.distance.pdist function. See also the
        ``dist_params`` parameter to specify and additional parameter for the
        distance function. Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each Evoked, a number indicating the item to which it belongs.
        When ``None``, each Evoked is assumed to belong to a different item.
        Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings
        (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels. Can
        also be the string values "all" to pick all channels, or "data" to pick
        data channels. ``None`` (default) will pick all MEG and EEG channels,
        excluding those maked as "bad".
    tmin : float | None
        When set, searchlight patches will only be generated from subsequent
        time points starting from this time point. This value is given in
        seconds. Defaults to ``None``, in which case patches are generated
        starting from the first time point.
    tmax : float | None
        When set, searchlight patches will only be generated up to and
        including this time point. This value is given in seconds. Defaults to
        ``None``, in which case patches are generated up to and including the
        last time point.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each searchlight patch.
    """
    times = evokeds[0].times
    for evoked in evokeds:
        if np.any(evoked.times != times):
            raise ValueError('Not all evokeds have the same time points.')

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = round(evokeds[0].info['sfreq'] * temporal_radius)


    # Normalize with the noise cov
    if noise_cov is not None:
        diag = spatial_radius is not None
        evokeds = [mne.whiten_evoked(evoked, noise_cov, diag=diag)
                   for evoked in evokeds]

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in evokeds[0].info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    picks = mne.io.pick._picks_to_idx(evokeds[0].info, picks, none='data')
    if len(picks) != len(set(picks)):
        raise ValueError("`picks` are not unique. Please remove duplicates.")
    samples_from, samples_to = _tmin_tmax_to_indices(times, tmin, tmax)

    # Compute the DSMs
    X = np.array([evoked.data for evoked in evokeds])
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius, sel_series=picks,
                          samples_from=samples_from, samples_to=samples_to)
    yield from dsm_array(X, patches, dist_metric=dist_metric,
                         dist_params=dist_params, y=y, n_folds=n_folds)


def dsm_epochs(epochs, noise_cov=None, spatial_radius=0.04,
               temporal_radius=0.1, dist_metric='correlation',
               dist_params=dict(), y=None, n_folds=1, picks=None,
               tmin=None, tmax=None, dropped_as_nan=False):
    """Generate DSMs in a searchlight pattern on epochs.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The brain activity during the epochs. The event codes are used to
        distinguish between items.
    noise_cov : mne.Covariance | None
        When specified, the data will by normalized using the noise covariance.
        This is recommended in all cases, but a hard requirement when the data
        contains sensors of different types. Defaults to None.
    spatial_radius : floats | None
        The spatial radius of the searchlight patch in meters. All sensors
        within this radius will belong to the searchlight patch. Set to None to
        only perform the searchlight over time, flattening across sensors.
        Defaults to 0.04.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to 0.1.
    dist_metric : str
        The metric to use to compute the DSM for the epochs. This can be any
        metric supported by the scipy.distance.pdist function. See also the
        ``epochs_dsm_params`` parameter to specify and additional parameter for
        the distance function. Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each Epoch, a number indicating the item to which it belongs.
        When ``None``, the event codes are used to differentiate between items.
        Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    picks : str | list | slice | None
        Channels to include. Slices and lists of integers will be interpreted
        as channel indices. In lists, channel *type* strings (e.g., ``['meg',
        'eeg']``) will pick channels of those types, channel *name* strings
        (e.g., ``['MEG0111', 'MEG2623']`` will pick the given channels. Can
        also be the string values "all" to pick all channels, or "data" to pick
        data channels. ``None`` (default) will pick all MEG and EEG channels,
        excluding those maked as "bad".
    tmin : float | None
        When set, searchlight patches will only be generated from subsequent
        time points starting from this time point. This value is given in
        seconds. Defaults to ``None``, in which case patches are generated
        starting from the first time point.
    tmax : float | None
        When set, searchlight patches will only be generated up to and
        including this time point. This value is given in seconds. Defaults to
        ``None``, in which case patches are generated up to and including the
        last time point.
    dropped_as_nan : bool
        When this is set to ``True``, the drop log will be used to inject NaN
        values in the DSMs at the locations where a bad epoch was dropped. This
        is useful to ensure the dimensions of the DSM are the same,
        irregardless of any bad epochs that were dropped. Make sure to use
        ``ignore_nan=True`` when using DSMs with NaNs in them during subsequent
        RSA computations. Defaults to ``False``.

        .. versionadded:: 0.8

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each searchlight patch.
    """
    if y is None:
        y = epochs.events[:, 2]

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = round(epochs.info['sfreq'] * temporal_radius)


    # # Normalize with the noise cov
    # if noise_cov is not None:
    #     diag = spatial_radius is not None
    #     evokeds = [mne.whiten_evoked(evoked, noise_cov, diag=diag)
    #                for evoked in evokeds]

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in epochs.info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    picks = mne.io.pick._picks_to_idx(epochs.info, picks, none='data')
    if len(picks) != len(set(picks)):
        raise ValueError("`picks` are not unique. Please remove duplicates.")
    samples_from, samples_to = _tmin_tmax_to_indices(epochs.times, tmin, tmax)

    # Compute the DSMs
    X = epochs.get_data()
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius,
                          sel_series=picks, samples_from=samples_from,
                          samples_to=samples_to)
    dsm_gen = dsm_array(X, patches, dist_metric=dist_metric,
                        dist_params=dist_params, y=y, n_folds=n_folds)
    if not dropped_as_nan or epochs.drop_log_stats() == 0:
        yield from dsm_gen
    else:
        nan_locations = [i for i, reason in enumerate(epochs.drop_log)
                         if len(reason) > 0]
        for dsm in dsm_gen:
            dsm = distance.squareform(dsm)
            dsm = np.insert(dsm, nan_locations, np.NaN, axis=0)
            dsm = np.insert(dsm, nan_locations, np.NaN, axis=1)
            # Can't use squareform to convert back due to the NaNs.
            yield dsm[np.triu_indices(len(dsm), 1)]


def _tmin_tmax_to_indices(times, tmin, tmax):
    """Convert tmin tmax parameters to an array of sample indices."""
    if tmin is None:
        samples_from = 0
    else:
        samples_from = np.searchsorted(times, tmin)
    if tmax is None:
        samples_to = len(times)
    else:
        samples_to = np.searchsorted(times, tmax)
    if samples_from > samples_to:
        raise ValueError(f'Invalid time range: {tmin} to {tmax}')
    return samples_from, samples_to


def _construct_tmin(times, samples_from, samples_to, temporal_radius):
    if temporal_radius is None:
        return times[(samples_from + samples_to) // 2]
    else:
        return times[max(temporal_radius, samples_from)]

def _square_to_condensed(i, j, n):
    if i < j:
        i, j = j, i
    return n * j - j * (j+1) // 2 + i - 1 - j
