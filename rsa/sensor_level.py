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

from .searchlight import (_temporal_radius_to_samples, _get_time_patch_centers,
                          _rsa_searchlight)


def rsa_evokeds(evokeds, model, evoked_dsm_metric='correlation',
                model_dsm_metric='correlation', rsa_metric='spearman',
                spatial_radius=0.04, temporal_radius=0.1, break_after=-1,
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
    spatial_radius : float
        The spatial radius of the searchlight patch in meters. All sensors
        within this radius will belong to the searchlight patch.
        Defaults to 0.04.
    temporal_radius : float
        The temporal radius of the searchlight patch in seconds.
        Defaults to 0.1.
    break_after : int
        Abort the computation after this many steps. Useful for debugging.
        Defaults to -1 which means to perform the computation until the end.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa : Evoked
        The correlation values for each searchlight patch.
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

    # Be careful with the default value for model_dsm_metric
    if model_dsm_metric in ['correlation', 'cosine'] and n_features == 1:
        raise ValueError("There is only a single model feature, so "
                         "'correlation' or 'cosine' can not be used as "
                         "model_dsm_metric. Consider using 'euclidean' "
                         "instead.")

    # Convert the temporal radius to samples
    temporal_radius = round(evokeds[0].info['sfreq'] * temporal_radius)

    if temporal_radius < 1:
        raise ValueError('Temporal radius is less than one sample.')

    # Compute the distances between the sensors
    locs = np.vstack([ch['loc'][:3] for ch in evokeds[0].info['chs']])
    dist = distance.squareform(distance.pdist(locs))

    # Construct a big array containing all brain data
    X = np.array([evoked.data for evoked in evokeds])

    # Perform the RSA
    rsa = _rsa_searchlight(X, model, dist, evoked_dsm_metric, model_dsm_metric,
                           rsa_metric, spatial_radius, temporal_radius,
                           break_after, n_jobs, verbose)

    # Pack the result in an Evoked object
    first_ind = _get_time_patch_centers(X.shape[1], temporal_radius)[0]
    return mne.EvokedArray(rsa, evokeds[0].info, tmin=times[first_ind],
                           comment='RSA', nave=len(evokeds))
