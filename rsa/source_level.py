# encoding: utf-8
"""
Module implementing representational similarity analysis (RSA) at the source
level.

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
from scipy.linalg import block_diag

from .searchlight import (_temporal_radius_to_samples, _get_time_patch_centers,
                          _rsa_searchlight)


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

    Returns
    -------
    stc : SourceEstimate
        The correlation values for each searchlight patch.
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
