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

from .rsa import (_get_time_patch_centers, rsa_spattemp, rsa_spat, rsa_temp,
                  compute_dsm, rsa)


def rsa_evokeds(evokeds, model, noise_cov=None, spatial_radius=0.04,
                temporal_radius=0.1, evoked_dsm_metric='correlation',
                model_dsm_metric='correlation', rsa_metric='spearman',
                n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern on evokeds. The inputs are:

    1) a list of Evoked objects that hold the evoked data for each item in the
       analysis.
    2) an item x features matrix that holds the model features for each item.
       The model can be other brain data, a computer model, norm data, etc.

    The output is an Evoked  object where the "signal" at each sensor is
    the RSA, computed using all surrounding sensors.

    Parameters
    ----------
    evokeds : list of mne.Evoked
        For each item, the evoked brain activity.
    model : ndarray, shape (n_items, n_features)
        For each item, the model features corresponding to the item.
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
        'correlation'.
    model_dsm_metric : str
        The metric to use to compute the DSM for the model features. This can
        be any metric supported by the scipy.distance.pdist function. Defaults
        to 'correlation'. Note that if the model only defines a few features,
        'euclidean' may be more appropriate.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs. This can either be
        'spearman' correlation or 'pearson' correlation.
        Defaults to 'spearman'.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa : Evoked
        The correlation values for each searchlight patch. When spatial_radius
        is set to None, there will only be one sensor. When temporal_radius is
        set to None, there will only be one time point.
    """
    # Check for compatibility of the source estimated and the model features
    n_items, n_features = model.shape
    if len(evokeds) != n_items:
        raise ValueError('The number of evokeds (%d) should be equal to the '
                         'number of items (%d).' % (len(evokeds), n_items))

    times = evokeds[0].times
    for evoked in evokeds:
        if np.any(evoked.times != times):
            raise ValueError('Not all evokeds have the same time points.')

    dsm_Y = compute_dsm(model, model_dsm_metric)

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = round(evokeds[0].info['sfreq'] * temporal_radius)
        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    # Construct a big array containing all brain data
    X = np.array([evoked.data for evoked in evokeds])

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in evokeds[0].info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    # Perform the RSA
    if spatial_radius is not None and temporal_radius is not None:
        data = rsa_spattemp(X, dsm_Y, dist, spatial_radius, temporal_radius,
                            evoked_dsm_metric, model_dsm_metric, rsa_metric,
                            n_jobs, verbose)
    elif spatial_radius is not None:
        data = rsa_spat(X, dsm_Y, dist, spatial_radius, evoked_dsm_metric,
                        model_dsm_metric, rsa_metric, n_jobs, verbose)
    elif temporal_radius is not None:
        data = rsa_temp(X, dsm_Y, evoked_dsm_metric, model_dsm_metric,
                        rsa_metric, n_jobs, verbose)
    else:
        data = rsa(X, dsm_Y, evoked_dsm_metric, model_dsm_metric,
                   rsa_metric, n_jobs, verbose)

    # Pack the result in an Evoked object
    if temporal_radius is not None:
        first_ind = _get_time_patch_centers(X.shape[1], temporal_radius)[0]
        tmin = times[first_ind]
    else:
        tmin = 0
    if spatial_radius is not None:
        info = evokeds[0].info
    else:
        info = mne.create_info(['rsa'], evoked[0].info['sfreq'])
    return mne.EvokedArray(data, info, tmin, comment='RSA', nave=len(evokeds))
