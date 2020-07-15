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

from warnings import warn
import numpy as np
import mne
from scipy.linalg import block_diag

from .dsm import _n_items_from_dsm, dsm_array
from .rsa import rsa_array
from .searchlight import searchlight
from .sensor_level import _tmin_tmax_to_indices, _construct_tmin


def rsa_source_level(stcs, dsm_model, src, spatial_radius=0.04,
                     temporal_radius=0.1, stc_dsm_metric='correlation',
                     stc_dsm_params=dict(), rsa_metric='spearman', y=None,
                     n_folds=1, sel_vertices=None, tmin=None, tmax=None,
                     n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern across the source space.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computed for a patch surrounding the source point.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate | list of mne.VolSourceEstimate
        For each item, a source estimate for the brain activity.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`compute_dsm` to compute DSMs.
    src : instance of mne.SourceSpaces
        The source space used by the source estimates specified in the `stcs`
        parameter.
    spatial_radius : float | None
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
        for the distance function. Defaults to 'correlation'.
    stc_dsm_params : dict
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
    y : ndarray of int, shape (n_items,) | None
        For each source estimate, a number indicating the item to which it
        belongs. When ``None``, each source estimate is assumed to belong to a
        different item. Defaults to ``None``.
    n_folds : int | None
        Number of folds to use when using cross-validation to compute the
        evoked DSM metric. Specify ``None``, to use the maximum number of folds
        possible, given the data.
        Defaults to 1 (no cross-validation).
    sel_vertices : list of int | None
        When set, searchlight patches will only be generated for the subset of
        vertices/voxels with the given indices. Defaults to ``None``, in which
        case patches for all vertices/voxels are generated.
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
    stc : SourceEstimate | VolSourceEstimate | list of SourceEstimate | list of VolSourceEstimate
        The correlation values for each searchlight patch. When spatial_radius
        is set to None, there will only be one vertex. When temporal_radius is
        set to None, there will only be one time point. When multiple models
        have been supplied, a list will be returned containing the RSA results
        for each model.

    See Also
    --------
    compute_dsm
    """  # noqa E501
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

    _check_compatible(stcs, src)
    dist = _get_distance_matrix(src, dist_lim=spatial_radius, n_jobs=n_jobs)

    if temporal_radius is not None:
        # Convert the temporal radius to samples
        temporal_radius = int(temporal_radius // stcs[0].tstep)

        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    sel_samples = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    # Perform the RSA
    X = np.array([stc.data for stc in stcs])
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius,
                          sel_series=sel_vertices, sel_samples=sel_samples)
    data = rsa_array(X, dsm_model, patches, data_dsm_metric=stc_dsm_metric,
                     data_dsm_params=stc_dsm_params, rsa_metric=rsa_metric,
                     y=y, n_folds=n_folds, n_jobs=n_jobs, verbose=verbose)

    # Pack the result in a SourceEstimate object
    if spatial_radius is not None:
        vertices = stcs[0].vertices
    else:
        if src.kind == 'volume':
            vertices = [np.array([1])]
        else:
            vertices = [np.array([1]), np.array([])]
    tmin = _construct_tmin(stcs[0].times, sel_samples, temporal_radius)
    tstep = stcs[0].tstep

    if one_model:
        if src.kind == 'volume':
            return mne.VolSourceEstimate(data, vertices, tmin, tstep,
                                         subject=stcs[0].subject)
        else:
            return mne.SourceEstimate(data, vertices, tmin, tstep,
                                      subject=stcs[0].subject)
    else:
        if src.kind == 'volume':
            return [mne.VolSourceEstimate(data[:, :, i], vertices, tmin, tstep,
                                          subject=stcs[0].subject)
                    for i in range(data.shape[-1])]
        else:
            return [mne.SourceEstimate(data[:, :, i], vertices, tmin, tstep,
                                       subject=stcs[0].subject)
                    for i in range(data.shape[-1])]


def dsm_source_level(stcs, src, spatial_radius=0.04, temporal_radius=0.1,
                     dist_metric='sqeuclidean', dist_params=dict(), y=None,
                     n_folds=None, sel_vertices=None, tmin=None, tmax=None,
                     n_jobs=1):
    """Generate DSMs in a searchlight pattern across the cortex.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computed for a patch surrounding the source point.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate | list of mne.VolSourceEstimate
        For each item, a source estimate for the brain activity.
    src : instance of mne.SourceSpaces
        The source space used by the source estimates specified in the `stcs`
        parameter.
    spatial_radius : float | None
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch. Set to
        None to only perform the searchlight over time, flattening across
        sensors. Defaults to 0.04.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in seconds. Set to None to
        only perform the searchlight over sensors, flattening across time.
        Defaults to 0.1.
    dist_metric : str
        The metric to use to compute the DSM for the source estimates. This can
        be any metric supported by the scipy.distance.pdist function. See also
        the ``stc_dsm_params`` parameter to specify and additional parameter
        for the distance function. Defaults to 'sqeuclidean'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each source estimate, a number indicating the item to which it
        belongs. When ``None``, each source estimate is assumed to belong to a
        different item. Defaults to ``None``.
    n_folds : int | None
        Number of folds to use when using cross-validation to compute the
        evoked DSM metric.  Defaults to ``None``, which means the maximum
        number of folds possible, given the data.
    sel_vertices : list of int | None
        When set, searchlight patches will only be generated for the subset of
        vertices/voxels with the given indices. Defaults to ``None``, in which
        case patches for all vertices/voxels are generated.
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
        The number of processes (=number of CPU cores) to use for the
        source-to-source distance computation. Specify -1 to use all available
        cores. Defaults to 1.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each searchlight patch.
    """
    _check_compatible(stcs, src)
    dist = _get_distance_matrix(src, dist_lim=spatial_radius, n_jobs=n_jobs)

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = int(temporal_radius // stcs[0].tstep)

        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    sel_samples = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    X = np.array([stc.data for stc in stcs])
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius,
                          sel_series=sel_vertices, sel_samples=sel_samples)
    yield from dsm_array(X, patches, dist_metric=dist_metric,
                         dist_params=dist_params, y=y, n_folds=n_folds)


def _check_compatible(stcs, src):
    """Check for compatibility of the source estimates and source space."""
    if src.kind == 'volume' and not isinstance(stcs[0], mne.VolSourceEstimate):
        raise ValueError(f'Volume source estimates provided, but not a volume '
                         f'source space (src.kind={src.kind}).')

    for stc in stcs:
        if src.kind == 'volume':
            if np.any(stc.vertices != src[0]['vertno']):
                raise ValueError('Not all source estimates have the same '
                                 'vertices.')
        else:
            for src_hemi, stc_hemi_vertno in zip(src, stcs[0].vertices):
                if np.any(stc_hemi_vertno != src_hemi['vertno']):
                    raise ValueError('Not all source estimates have the same '
                                     'vertices.')

    times = stcs[0].times
    for stc in stcs:
        if np.any(stc.times != times):
            raise ValueError('Not all source estimates have the same '
                             'time points.')


def _get_distance_matrix(src, dist_lim, n_jobs=1):
    """Get vertex-to-vertex distance matrix from source space.

    During inverse computation, the source space was downsampled (i.e. using
    ico4). Construct vertex-to-vertex distance matrices using only the
    vertices that are defined in the source solution.

    Parameters
    ----------
    src : mne.SourceSpaces
        The source space to get the distance matrix for.
    dist_lim : float
        Maximum distance required. We don't care about distances beyond this
        maximum.
    n_jobs : int
        Number of CPU cores to use if distance computation is necessary.
        Defaults to 1.

    Returns
    -------
    dist : ndarray (n_vertices, n_vertices)
        The vertex-to-vertex distance matrix.
    """
    dist = []

    # Check if distances have been pre-computed in the given source space. Give
    # a warning if the pre-computed distances may have had a too limited
    # dist_lim setting.
    needs_distance_computation = False
    for hemi in src:
        if 'dist' not in hemi or hemi['dist'] is None:
            needs_distance_computation = True
        else:
            if hemi['dist_limit'][0] < dist_lim:
                warn(f'Source space has pre-computed distances, but all '
                     f'distances are smaller than the searchlight radius '
                     f'({dist_lim}). You may want to consider recomputing '
                     f'the source space distances using the '
                     f'mne.add_source_space_distances function.')

    if needs_distance_computation:
        if dist_lim is None:
            dist_lim = np.inf
        if src.kind == 'volume':
            src = _add_volume_source_space_distances(src, dist_lim)
        else:
            src = mne.add_source_space_distances(src, dist_lim, n_jobs=n_jobs)

    for hemi in src:
        inuse = np.flatnonzero(hemi['inuse'])

        dist.append(hemi['dist'][np.ix_(inuse, inuse)].toarray())

    # Collect the distances in a single matrix
    dist = block_diag(*dist)
    dist[dist == 0] = np.inf  # Across hemisphere distance is infinity
    dist.flat[::dist.shape[0] + 1] = 0  # Distance to yourself is zero

    return dist


def _add_volume_source_space_distances(src, dist_limit):
    """Compute the distance between voxels in a volume source space.

    Operates in-place!

    Parameters
    ----------
    src : instance of mne.SourceSpaces
        The volume source space to compute the voxel-wise distances for.
    dist_limit : float
        The maximum distance (in meters) to consider. Voxels that are further
        apart than this distance will have a distance of infinity. Use this to
        reduce computation time.

    Returns
    -------
    src : instance of mne.SourceSpaces
        The volume source space, now with the 'dist' and 'dist_limit' fields
        set.
    """
    # Lazy import to not have to load the huge scipy module every time mne_rsa
    # get's loaded.
    from scipy.sparse import csr_matrix
    assert src.kind == 'volume'
    n_sources = src[0]['np']
    neighbors = np.array(src[0]['neighbor_vert'])
    row, col = np.nonzero(neighbors != -1)
    col = neighbors[(row, col)]
    con = np.linalg.norm(src[0]['rr'][row, :] - src[0]['rr'][col, :], axis=1)
    con_matrix = csr_matrix((con, (row, col)), shape=(n_sources, n_sources))
    dist = mne.source_space._do_src_distances(con_matrix, src[0]['vertno'],
                                              np.arange(src[0]['nuse']),
                                              dist_limit)[0]
    d = dist.ravel()  # already float32
    idx = d > 0
    d = d[idx]
    i, j = np.meshgrid(src[0]['vertno'], src[0]['vertno'])
    i = i.ravel()[idx]
    j = j.ravel()[idx]
    src[0]['dist'] = csr_matrix((d, (i, j)), shape=(n_sources, n_sources))
    src[0]['dist_limit'] = np.array([dist_limit], np.float32)
    return src
