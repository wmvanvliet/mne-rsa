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

from .dsm import _get_time_patch_centers, _n_items_from_dsm
from .rsa import rsa_array


def rsa_source_level(stcs, dsm_model, src, spatial_radius=0.04,
                     temporal_radius=0.1, stc_dsm_metric='sqeuclidean',
                     stc_dsm_params=dict(), rsa_metric='spearman', y=None,
                     n_folds=None, n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern across the cortex.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computed for a patch surrounding the source point.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate
        For each item, a source estimate for the brain activity.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`rsa.compute_dsm` to compute DSMs.
    src : instance of mne.SourceSpaces
        The source space used by the source estimates specified in the `stcs`
        parameter.
    spatial_radius : floats | None
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch. Set to
        None to only perform the searchlight over time, flattening across
        sensors. Defaults to 0.04.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to 0.1.
    stc_dsm_metric : str
        The metric to use to compute the DSM for the source estimates. This can
        be any metric supported by the scipy.distance.pdist function. See also
        the ``stc_dsm_params`` parameter to specify and additional parameter
        for the distance function. Defaults to 'sqeuclidean'.
    stc_dsm_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    rsa_metric : 'spearman' | 'pearson'
        The metric to use to compare the stc and model DSMs. This can either be
        'spearman' correlation or 'pearson' correlation.
        Defaults to 'spearman'.
    y : ndarray of int, shape (n_items,) | None
        For each source estimate, a number indicating the item to which it
        belongs. When ``None``, each source estimate is assumed to belong to a
        different item. Defaults to ``None``.
    n_folds : int | None
        Number of folds to use when using cross-validation to compute the
        evoked DSM metric.  Defaults to ``None``, which means the maximum
        number of folds possible, given the data.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    stc : SourceEstimate | list of SourceEstimate
        The correlation values for each searchlight patch. When spatial_radius
        is set to None, there will only be one vertex. When temporal_radius is
        set to None, there will only be one time point. When multiple models
        have been supplied, a list will be returned containing the RSA results
        for each model.

    See Also
    --------
    compute_dsm
    """
    # Check for compatibility of the source estimated and the model features
    one_model = type(dsm_model) is np.ndarray
    if one_model:
        dsm_model = [dsm_model]

    # Check for compatibility of the evokeds and the model features
    for dsm in dsm_model:
        n_items = _n_items_from_dsm(dsm)
        if len(stcs) != n_items and y is None:
            raise ValueError(
                'The number of source estimates (%d) should be equal to the '
                'number of items in `dsm_model` (%d). Alternatively, use '
                'the `y` parameter to assign evokeds to items.'
                % (len(stcs), n_items))
        if y is not None and len(np.unique(y)) != n_items:
            raise ValueError(
                'The number of items in `dsm_model` (%d) does not match '
                'the number of items encoded in the `y` matrix (%d).'
                % (n_items, len(np.unique(y))))

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

    if temporal_radius is not None:
        # Convert the temporal radius to samples
        temporal_radius = int(temporal_radius // stcs[0].tstep)

        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    # During inverse computation, the source space was downsampled (i.e. using
    # ico4). Construct vertex-to-vertex distance matrices using only the
    # vertices that are defined in the source solution.
    dist = []
    for hemi in [0, 1]:
        inuse = np.flatnonzero(src[hemi]['inuse'])
        dist.append(src[hemi]['dist'][np.ix_(inuse, inuse)].toarray())

    # Collect the distances in a single matrix
    dist = block_diag(*dist)
    dist[dist == 0] = np.inf  # Across hemisphere distance is infinity
    dist[::dist.shape[0] + 1] = 0  # Distance to yourself is zero

    # Construct a big array containing all brain data
    X = np.array([stc.data for stc in stcs])

    # Perform the RSA
    data = rsa_array(X, dsm_model, dist=dist, spatial_radius=spatial_radius,
                     temporal_radius=temporal_radius,
                     data_dsm_metric=stc_dsm_metric,
                     data_dsm_params=stc_dsm_params, rsa_metric=rsa_metric,
                     y=y, n_folds=n_folds, n_jobs=n_jobs, verbose=verbose)

    # Pack the result in a SourceEstimate object
    if temporal_radius is not None:
        first_ind = _get_time_patch_centers(X.shape[-1], temporal_radius)[0]
        tmin = times[first_ind]
        tstep = stcs[0].tstep
    else:
        tmin = 0
        tstep = 1
    if spatial_radius is not None:
        vertices = [lh_verts, rh_verts]
    else:
        vertices = [np.array([1]), np.array([])]

    if one_model:
        return mne.SourceEstimate(data[:, :, 0], vertices, tmin, tstep,
                                  subject=stcs[0].subject)
    else:
        return [mne.SourceEstimate(data[:, :, i], vertices, tmin, tstep,
                                   subject=stcs[0].subject)
                for i in range(data.shape[-1])]
