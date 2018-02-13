# encoding: utf-8
"""
Module implementing representational similarity analysis (RSA). See:

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational
similarity analysis - connecting the branches of systems neuroscience.
Frontiers in Systems Neuroscience, 2(November), 4.
https://doi.org/10.3389/neuro.06.004.2008

Authors
-------
Marijn van Vliet <marijn.vanvliet@aalto.fi>
Annika Hultén <annika.hulten@aalto.fi>
Ossi Lehtonen <ossi.lehtonen@aalto.fi>
"""

import numpy as np
import mne
from scipy import stats
from scipy.spatial import distance
from scipy.linalg import block_diag


def rsa_source_level(stcs, model, src, stc_dsm_metric='correlation',
                     model_dsm_metric='correlation', rsa_metric='spearman',
                     spatial_radius=0.04, temporal_radius=0.1, break_after=-1,
                     n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern across the cortex. The inputs are:

    1) a list of SourceEstimate objects that hold the source estimate for each
       item in the analysis.
    2) an item x features matrix that holds the model features for each item.
       The model can be other brain data, a computer model, norm data, etc.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computes for a patch surrounding the source point.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate
        For each item, a source estimate for the brain activity.
    model : ndarray, shape (n_items, n_features)
        For each item, the model features corresponding to the item.
    src : instance of mne.SourceSpaces
        The source space used by the source estimates specified in the `stcs`
        parameter.
    stc_dsm_metric : str
        The metric to use to compute the DSM for the source estimates. This can
        be any metric supported by the scipy.distance.pdist function. Defaults
        to 'correlation'.
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
        The spatial radius of the searchlight patch in meters.
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
    """
    # Check for compatibility of the source estimated and the model features
    n_items, n_features = model.shape
    if len(stcs) != n_items:
        raise ValueError('The number of source estimates (%d) should be equal '
                         'to the number of items (%d).' % (len(stcs), n_items))

    # Check for compatibility of the source estimates and source space
    lh_verts, rh_verts = stcs[0].vertices
    for stc in stcs:
        if (np.any(stc.vertices[0] != lh_verts) or
                np.any(stc.vertices[1] != rh_verts)):
            raise ValueError('Not all source estimates have the same '
                             'vertices.')
    if (np.any(src[0]['vertno'] != lh_verts) or
            np.any(src[1]['vertno'] != rh_verts)):
        raise ValueError('The source space is not defined for the same '
                         'vertices as the source estimates.')

    times = stcs[0].times
    for stc in stcs:
        if np.any(stc.times != times):
            raise ValueError('Not all source estimates have the same '
                             'time points.')

    # Be careful with the default value for model_dsm_metric
    if model_dsm_metric in ['correlation', 'cosine'] and n_features == 1:
        raise ValueError("There is only a single model feature, so "
                         "'correlation' or 'cosine' can not be used as "
                         "model_dsm_metric. Consider using 'euclidean' "
                         "instead.")

    # Convert the temporal radius to samples
    temporal_radius = _temporal_radius_to_samples(stcs[0].tstep,
                                                  temporal_radius)

    if temporal_radius < 1:
        raise ValueError('Temporal radius is less than one sample.')

    # During inverse computation, the source space was downsampled (i.e. using
    # ico4). Construct vertex-to-vertex distance matrices using only the
    # vertices that are defined in the source solution.
    dist = []
    for hemi in [0, 1]:
        inuse = np.flatnonzero(src[0]['inuse'])
        dist.append(src[hemi]['dist'][np.ix_(inuse, inuse)].toarray())

    # Collect the distances in a single matrix
    dist = block_diag(*dist)
    dist[dist == 0] = np.inf  # Across hemisphere distance is infinity
    dist[::dist.shape[0] + 1] = 0  # Distance to yourself is zero

    # Construct a big array containing all brain data
    X = np.array([stc.data for stc in stcs])

    # Perform the RSA
    rsa = _rsa_searchlight(X, model, dist, stc_dsm_metric, model_dsm_metric,
                           rsa_metric, spatial_radius, temporal_radius,
                           break_after, n_jobs, verbose)

    # Pack the result in a SourceEstimate object
    first_ind = _get_time_patch_centers(X.shape[1], temporal_radius)[0]
    return mne.SourceEstimate(rsa, vertices=[lh_verts, rh_verts],
                              tmin=times[first_ind], tstep=stcs[0].tstep)


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
    """Convert the temporal radius from seconds to samples.

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
