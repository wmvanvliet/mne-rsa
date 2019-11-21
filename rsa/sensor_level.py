# encoding: utf-8
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

from .dsm import _get_time_patch_centers, _n_items_from_dsm
from .rsa import rsa_array


def rsa_evokeds(evokeds, dsm_model, noise_cov=None, spatial_radius=0.04,
                temporal_radius=0.1, evoked_dsm_metric='sqeuclidean',
                evoked_dsm_params=None, rsa_metric='spearman', y=None,
                n_folds=None, n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern on evokeds.

    The output is an Evoked object where the "signal" at each sensor is
    the RSA, computed using all surrounding sensors.

    Parameters
    ----------
    evokeds : list of mne.Evoked
        The evoked brain activity for each item. If you have more than one
        Evoked object per item (i.e. repetitions), you can use the ``y``
        parameter to assign evokeds to items.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`rsa.compute_dsm` to compute DSMs.
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
    evoked_dsm_metric : str
        The metric to use to compute the DSM for the evokeds. This can be any
        metric supported by the scipy.distance.pdist function. Defaults to
        'sqeuclidean'.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs. This can either be
        'spearman' correlation or 'pearson' correlation.
        Defaults to 'spearman'.
    y : ndarray of int, shape (n_items,) | None
        For each Evoked, a number indicating the item to which it belongs.
        When ``None``, each Evoked is assumed to belong to a different item.
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
        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    # Normalize with the noise cov
    if noise_cov is not None:
        diag = spatial_radius is not None
        evokeds = [mne.whiten_evoked(evoked, noise_cov, diag=diag)
                   for evoked in evokeds]

    # Construct a big array containing all brain data
    X = np.array([evoked.data for evoked in evokeds])

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in evokeds[0].info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    # Perform the RSA
    data = rsa_array(X, dsm_model, dist=dist, spatial_radius=spatial_radius,
                     temporal_radius=temporal_radius,
                     data_dsm_metric=evoked_dsm_metric,
                     data_dsm_params=evoked_dsm_params, rsa_metric=rsa_metric,
                     y=y, n_folds=n_folds, n_jobs=n_jobs, verbose=verbose)

    # Pack the result in an Evoked object
    if temporal_radius is not None:
        first_ind = _get_time_patch_centers(X.shape[-1], temporal_radius)[0]
        tmin = times[first_ind]
    else:
        tmin = 0
    if spatial_radius is not None:
        info = evokeds[0].info
    else:
        info = mne.create_info(['rsa'], evokeds[0].info['sfreq'])

    if one_model:
        return mne.EvokedArray(data[:, :, 0], info, tmin, comment='RSA',
                               nave=len(evokeds))
    else:
        return [mne.EvokedArray(data[:, :, i], info, tmin, comment='RSA',
                                nave=len(evokeds))
                for i in range(data.shape[-1])]


def rsa_epochs(epochs, dsm_model, noise_cov=None, spatial_radius=0.04,
               temporal_radius=0.1, epochs_dsm_metric='sqeuclidean',
               epochs_dsm_params=None, rsa_metric='spearman', y=None,
               n_folds=None, n_jobs=1, verbose=False):
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

        Use :func:`rsa.compute_dsm` to compute DSMs.
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
    evoked_dsm_metric : str
        The metric to use to compute the DSM for the evokeds. This can be any
        metric supported by the scipy.distance.pdist function. Defaults to
        'sqeuclidean'.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs. This can either be
        'spearman' correlation or 'pearson' correlation.
        Defaults to 'spearman'.
    y : ndarray of int, shape (n_items,) | None
        For each Epoch, a number indicating the item to which it belongs.
        When ``None``, the event codes are used to differentiate between items.
        Defaults to ``None``.
    n_folds : int | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter, or the event
        codes if ``y`` is not specified. Specify -1 to use the maximum number
        of folds possible, given the data.
        Defaults to 1 (no cross-validation).
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

    if y is None:
        y_source = 'Epoch object'
        y = epochs.events[:, 2]
    else:
        y_source = '`y` matrix'

    # Check for compatibility of the evokeds and the model features
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
        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    # # Normalize with the noise cov
    # if noise_cov is not None:
    #     diag = spatial_radius is not None
    #     evokeds = [mne.whiten_evoked(evoked, noise_cov, diag=diag)
    #                for evoked in evokeds]

    # Construct a big array containing all brain data
    X = epochs.get_data()

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in epochs.info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    # Perform the RSA
    data = rsa_array(X, dsm_model, dist=dist, spatial_radius=spatial_radius,
                     temporal_radius=temporal_radius,
                     data_dsm_metric=epochs_dsm_metric,
                     data_dsm_params=epochs_dsm_params, rsa_metric=rsa_metric,
                     y=y, n_folds=n_folds, n_jobs=n_jobs, verbose=verbose)

    # Pack the result in an Evoked object
    if temporal_radius is not None:
        first_ind = _get_time_patch_centers(X.shape[-1], temporal_radius)[0]
        tmin = epochs.times[first_ind]
    else:
        tmin = 0
    if spatial_radius is not None:
        info = epochs.info
    else:
        info = mne.create_info(['rsa'], epochs.info['sfreq'])

    if one_model:
        return mne.EvokedArray(data[:, :, 0], info, tmin, comment='RSA',
                               nave=len(epochs.event_id))
    else:
        return [mne.EvokedArray(data[:, :, i], info, tmin, comment='RSA',
                                nave=len(epochs.event_id))
                for i in range(data.shape[-1])]
