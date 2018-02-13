# encoding: utf-8
"""
Module implementing private functions to perform RSA in searchlight patches.


Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
Ossi Lehtonen <ossi.lehtonen@aalto.fi>
"""

import numpy as np
import mne
from scipy import stats
from scipy.spatial import distance


def _rsa_searchlight(X, Y, dist, stc_dsm_metric, model_dsm_metric, rsa_metric,
                     spatial_radius, temporal_radius, break_after, n_jobs,
                     verbose):
    """Compute spatio-temporal RSA using a searchlight pattern.

    This implements the actual computations and assumes all sanity checks have
    been performed earlier by the rsa_source_level and rsa_sensor_level
    functions.

    Parameters
    ----------
    X : ndarray, shape (n_items, n_series, n_times)
        The brain data in the form of an ndarray. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis.
    Y : array, shape (n_items, n_features)
        The model features.
    dist : ndarray, shape (n_series, n_series)
        The distance between the source points or sensors.
    stc_dsm_metric : str
        The metric to use to compute the DSM for the source estimates.
    model_dsm_metric : str
        The metric to use to compute the DSM for the model features.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters.
    temporal_radius : int
        The temporal radius of the searchlight patch in samples.
    break_after : int
        Abort the computation after this many steps. Useful for debugging.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores.
    verbose : bool
        Whether to display a progress bar.

    Returns
    -------
    results : ndarray, shape (n_series, n_times)
        The RSA correlation values for each spatio-temporal patch.
    """
    n_series, n_samples = X.shape[1:]

    # Compute the dissimilarity matrix of the model features
    dsm_Y = distance.pdist(Y, metric=model_dsm_metric)

    # Progress bar
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=n_series)

    def progress(sequence):
        step = 0
        for item in sequence:
            if verbose:
                pbar.update(1)
            if break_after >= 0 and step > break_after:
                break
            step += 1
            yield item
        if verbose:
            pbar.close()

    # Use multiple cores to do the RSA computation
    parallel, my_rsa_searchlight_step, _ = mne.parallel.parallel_func(
        _rsa_searchlight_patch, n_jobs, verbose=False)

    # Run RSA for each spatial patch
    results = []
    results += parallel(
        my_rsa_searchlight_step(
            X[:, np.flatnonzero(series_dist < spatial_radius), :],
            dsm_Y, stc_dsm_metric, rsa_metric, temporal_radius)
        for series_dist in progress(dist)
    )

    return np.array(results)


def _rsa_searchlight_patch(X, dsm_Y, stc_dsm_metric, rsa_metric,
                           temporal_radius):
    """Perform temporal RSA analysis for a single spatial patch.

    This function is called in a parallel fashion across all searchlight
    patches along the cortex. It computes the RSA across time.

    Parameters
    ---------
    X : ndarray, shape (n_items, n_series, n_times)
        The brain data in the spatial patch. The second dimension can
        either be source points for source-level analysis or channels for
        sensor-level analysis. Only the points/channels inside the spatial
        patch should be included.
    dsm_Y : ndarray, shape (n_items, n_items)
        The model DSM.
    stc_dsm_metric : str
        The metric to use to compute the DSM for the source estimates.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs.
    temporal_radius : int
        The temporal radius of the searchlight patch in samples.

    Returns
    -------
    results : ndarray, shape (n_temporal_patches,)
        The RSA correlation values for each temporal patch.
    """
    n_items, _, n_samples = X.shape

    # Iterate over the temporal patches
    results = []
    for sample in _get_time_patch_centers(n_samples, temporal_radius):
        x = X[:, :, sample - temporal_radius:sample + temporal_radius]
        x = x.reshape(n_items, -1)
        dsm_x = distance.pdist(x, metric=stc_dsm_metric)

        if rsa_metric == 'spearman':
            r, _ = stats.spearmanr(dsm_x, dsm_Y)
        else:
            r, _ = stats.pearsonr(dsm_x, dsm_Y)
        results.append(r)

    return np.array(results)


def _get_time_patch_centers(n_samples, temporal_radius):
    """Compute the centers for the temporal patches.

    Parameters
    ----------
    n_samples : int
        The total number of samples.
    temporal_radius : int
        The temporal radius of the patches in samples.

    Returns
    -------
    time_inds : list of int
        For each temporal patch, the time-index of the middle of the patch
    """
    return list(range(temporal_radius, n_samples - temporal_radius + 1))


def _temporal_radius_to_samples(tstep, temporal_radius):
    """Convert a temporal radius from seconds to samples.

    Parameters
    ----------
    tstep : float
        The time interval between two samples in seconds.
    temporal_radius : float
        The temporal radius of the patches in seconds.

    Returns
    -------
    temporal_radius : int
        The temporal radius in samples.
    """
    return int(temporal_radius // tstep)
