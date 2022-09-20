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
from mne.utils import logger
from scipy.linalg import block_diag
import nibabel as nib

from .dsm import _n_items_from_dsm, dsm_array
from .rsa import rsa_array
from .searchlight import searchlight
from .sensor_level import _tmin_tmax_to_indices, _construct_tmin


def rsa_stcs(stcs, dsm_model, src, spatial_radius=0.04, temporal_radius=0.1,
             stc_dsm_metric='correlation', stc_dsm_params=dict(),
             rsa_metric='spearman', ignore_nan=False, y=None, n_folds=1,
             sel_vertices=None, tmin=None, tmax=None, n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern on MNE-Python source estimates.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computed for a patch surrounding the source point. Source estimate
    objects can be either defined along a cortical surface (``SourceEstimate``
    objects) or volumetric (``VolSourceEstimate`` objects).  For surface source
    estimates, distances between vertices are measured in 2D space, namely as
    the length of the path along the surface from one vertex to another. For
    volume source estimates, distances are measured in 3D space as a straight
    line from one voxel to another.

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
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    y : ndarray of int, shape (n_items,) | None
        For each source estimate, a number indicating the item to which it
        belongs. When ``None``, each source estimate is assumed to belong to a
        different item. Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
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
    # Check for compatibility of the source estimates and the model features
    one_model = type(dsm_model) is np.ndarray
    if one_model:
        dsm_model = [dsm_model]

    # Check for compatibility of the stcs and the model features
    for dsm in dsm_model:
        n_items = _n_items_from_dsm(dsm)
        if len(stcs) != n_items and y is None:
            raise ValueError(
                'The number of source estimates (%d) should be equal to the '
                'number of items in `dsm_model` (%d). Alternatively, use '
                'the `y` parameter to assign source estimates to items.'
                % (len(stcs), n_items))
        if y is not None and len(np.unique(y)) != n_items:
            raise ValueError(
                'The number of items in `dsm_model` (%d) does not match '
                'the number of items encoded in the `y` matrix (%d).'
                % (n_items, len(np.unique(y))))

    _check_stcs_compatibility(stcs, src)
    if spatial_radius is not None:
        dist = _get_distance_matrix(src, dist_lim=spatial_radius,
                                    n_jobs=n_jobs)
    else:
        dist = None
    if temporal_radius is not None:
        # Convert the temporal radius to samples
        temporal_radius = int(temporal_radius // stcs[0].tstep)
        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    samples_from, samples_to = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    # Perform the RSA
    X = np.array([stc.data for stc in stcs])
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius,
                          sel_series=sel_vertices, samples_from=samples_from,
                          samples_to=samples_to)
    data = rsa_array(X, dsm_model, patches, data_dsm_metric=stc_dsm_metric,
                     data_dsm_params=stc_dsm_params, rsa_metric=rsa_metric,
                     ignore_nan=ignore_nan, y=y, n_folds=n_folds,
                     n_jobs=n_jobs, verbose=verbose)

    # Pack the result in a SourceEstimate object
    if spatial_radius is not None:
        vertices = stcs[0].vertices
        if sel_vertices is not None:
            vertices = vertices[sel_vertices]
    else:
        if src.kind == 'volume':
            vertices = [np.array([1])]
        else:
            vertices = [np.array([1]), np.array([])]
        data = data[np.newaxis, ...]
    tmin = _construct_tmin(stcs[0].times, samples_from, samples_to,
                           temporal_radius)
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
            return [mne.VolSourceEstimate(data[..., i], vertices, tmin, tstep,
                                          subject=stcs[0].subject)
                    for i in range(data.shape[-1])]
        else:
            return [mne.SourceEstimate(data[..., i], vertices, tmin, tstep,
                                       subject=stcs[0].subject)
                    for i in range(data.shape[-1])]


def dsm_stcs(stcs, src, spatial_radius=0.04, temporal_radius=0.1,
             dist_metric='sqeuclidean', dist_params=dict(), y=None,
             n_folds=None, sel_vertices=None, tmin=None, tmax=None, n_jobs=1,
             verbose=False):
    """Generate DSMs in a searchlight pattern on MNE-Python source estimates.

    DSMs are computed using a patch surrounding each source point. Source
    estimate objects can be either defined along a cortical surface
    (``SourceEstimate`` objects) or volumetric (``VolSourceEstimate`` objects).
    For surface source estimates, distances between vertices are measured in 2D
    space, namely as the length of the path along the surface from one vertex
    to another. For volume source estimates, distances are measured in 3D space
    as a straight line from one voxel to another.

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
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
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
        The number of processes (=number of CPU cores) to use for the
        source-to-source distance computation. Specify -1 to use all available
        cores. Defaults to 1.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each searchlight patch.
    """
    _check_stcs_compatibility(stcs, src)
    if spatial_radius is not None:
        dist = _get_distance_matrix(src, dist_lim=spatial_radius,
                                    n_jobs=n_jobs)
    else:
        dist = None

    # Convert the temporal radius to samples
    if temporal_radius is not None:
        temporal_radius = int(temporal_radius // stcs[0].tstep)

        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    samples_from, samples_to = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    X = np.array([stc.data for stc in stcs])
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=temporal_radius,
                          sel_series=sel_vertices, samples_from=samples_from,
                          samples_to=samples_to)
    yield from dsm_array(X, patches, dist_metric=dist_metric,
                         dist_params=dist_params, y=y, n_folds=n_folds)


def rsa_stcs_rois(stcs, dsm_model, src, rois, temporal_radius=0.1,
                  stc_dsm_metric='correlation', stc_dsm_params=dict(),
                  rsa_metric='spearman', ignore_nan=False, y=None, n_folds=1,
                  sel_vertices=None, tmin=None, tmax=None, n_jobs=1,
                  verbose=False):
    """Perform RSA for a list of ROIs using MNE-Python source estimates.

    The output is a source estimate where the "signal" at each source point is
    the RSA, computed for a patch surrounding the source point. Source estimate
    objects can be either defined along a cortical surface (``SourceEstimate``
    objects) or volumetric (``VolSourceEstimate`` objects).  For surface source
    estimates, distances between vertices are measured in 2D space, namely as
    the length of the path along the surface from one vertex to another. For
    volume source estimates, distances are measured in 3D space as a straight
    line from one voxel to another.

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
    rois : list of mne.Label
        The spatial regions of interest (ROIs) to compute the RSA for. This
        needs to be specified as a list of ``mne.Label`` objects, such as
        returned by ``mne.read_annotations``.
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
    ignore_nan : bool
        Whether to treat NaN's as missing values and ignore them when computing
        the distance metric. Defaults to ``False``.

        .. versionadded:: 0.8
    y : ndarray of int, shape (n_items,) | None
        For each source estimate, a number indicating the item to which it
        belongs. When ``None``, each source estimate is assumed to belong to a
        different item. Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
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
    data : ndarray, shape (n_rois, n_times) | list of ndarray
        The correlation values for each ROI. When temporal_radius is set to
        None, there will be time dimension. When multiple models have been
        supplied, a list will be returned containing the RSA results for each
        model.
    stc : SourceEstimate | list of SourceEstimate
        The correlation values for each ROI, backfilled into a full
        SourceEstimate object. Each vertex belonging to the same ROI will have
        the same values. When temporal_radius is set to None, there will only
        be one time point. When multiple models have been supplied, a list will
        be returned containing the RSA results for each model.

    See Also
    --------
    compute_dsm
    """  # noqa E501
    # Check for compatibility of the source estimates and the model features
    one_model = type(dsm_model) is np.ndarray
    if one_model:
        dsm_model = [dsm_model]

    # Check for compatibility of the stcs and the model features
    for dsm in dsm_model:
        n_items = _n_items_from_dsm(dsm)
        if len(stcs) != n_items and y is None:
            raise ValueError(
                'The number of source estimates (%d) should be equal to the '
                'number of items in `dsm_model` (%d). Alternatively, use '
                'the `y` parameter to assign source estimates to items.'
                % (len(stcs), n_items))
        if y is not None and len(np.unique(y)) != n_items:
            raise ValueError(
                'The number of items in `dsm_model` (%d) does not match '
                'the number of items encoded in the `y` matrix (%d).'
                % (n_items, len(np.unique(y))))

    _check_stcs_compatibility(stcs, src)

    if temporal_radius is not None:
        # Convert the temporal radius to samples
        temporal_radius = int(temporal_radius // stcs[0].tstep)

        if temporal_radius < 1:
            raise ValueError('Temporal radius is less than one sample.')

    samples_from, samples_to = _tmin_tmax_to_indices(stcs[0].times, tmin, tmax)

    # Convert the labels to data indices
    roi_inds = list()
    for roi in rois:
        roi = roi.copy().restrict(src)
        if roi.hemi == 'lh':
            roi_ind = np.searchsorted(src[0]['vertno'], roi.vertices)
        else:
            roi_ind = np.searchsorted(src[1]['vertno'], roi.vertices)
            roi_ind += src[0]['nuse']
        roi_inds.append(roi_ind)

    # Perform the RSA
    X = np.array([stc.data for stc in stcs])
    patches = searchlight(X.shape, spatial_radius=roi_inds,
                          temporal_radius=temporal_radius,
                          sel_series=sel_vertices, samples_from=samples_from,
                          samples_to=samples_to)
    data = rsa_array(X, dsm_model, patches, data_dsm_metric=stc_dsm_metric,
                     data_dsm_params=stc_dsm_params, rsa_metric=rsa_metric,
                     ignore_nan=ignore_nan, y=y, n_folds=n_folds,
                     n_jobs=n_jobs, verbose=verbose)

    # Pack the result in SourceEstimate objects
    vertices = stcs[0].vertices
    subject = stcs[0].subject
    if sel_vertices is not None:
        vertices = vertices[sel_vertices]
    tmin = _construct_tmin(stcs[0].times, samples_from, samples_to,
                           temporal_radius)
    tstep = stcs[0].tstep
    if one_model:
        stc = backfill_stc_from_rois(data, rois, src, tmin=tmin, tstep=tstep,
                                     subject=subject)
    else:
        stc = [backfill_stc_from_rois(data[..., i], rois, src, tmin=tmin,
                                      tstep=tstep, subject=subject)
               for i in range(data.shape[-1])]

    return data, stc


def rsa_nifti(image, dsm_model, spatial_radius=0.01,
              image_dsm_metric='correlation', image_dsm_params=dict(),
              rsa_metric='spearman', ignore_nan=False, y=None, n_folds=1,
              roi_mask=None, brain_mask=None, n_jobs=1, verbose=False):
    """Perform RSA in a searchlight pattern on Nibabel Nifti-like images.

    The output is a 3D Nifti image where the data at each voxel is is
    the RSA, computed for a patch surrounding the voxel.

    .. versionadded:: 0.4

    Parameters
    ----------
    image : 4D Nifti-like image
        The Nitfi image data. The 4th dimension must contain the images
        for each item.
    dsm_model : ndarray, shape (n, n) | (n * (n - 1) // 2,) | list of ndarray
        The model DSM, see :func:`compute_dsm`. For efficiency, you can give it
        in condensed form, meaning only the upper triangle of the matrix as a
        vector. See :func:`scipy.spatial.distance.squareform`. To perform RSA
        against multiple models at the same time, supply a list of model DSMs.

        Use :func:`compute_dsm` to compute DSMs.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch.
        Defaults to 0.01.
    image_dsm_metric : str
        The metric to use to compute the DSM for the data. This can be
        any metric supported by the scipy.distance.pdist function. See also the
        ``image_dsm_params`` parameter to specify and additional parameter for
        the distance function. Defaults to 'correlation'.
    image_dsm_params : dict
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
        For each source estimate, a number indicating the item to which it
        belongs. When ``None``, each source estimate is assumed to belong to a
        different item. Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    roi_mask : 3D Nifti-like image | None
        When set, searchlight patches will only be generated for the subset of
        voxels with non-zero values in the given mask. This is useful for
        restricting the analysis to a region of interest (ROI). Note that while
        the center of the patches are all within the ROI, the patch itself may
        extend beyond the ROI boundaries.
        Defaults to ``None``, in which case patches for all voxels are
        generated.
    brain_mask : 3D Nifti-like image | None
        When set, searchlight patches are restricted to only contain voxels
        with non-zero values in the given mask. This is useful for make sure
        only information from inside the brain is used. In contrast to the
        `roi_mask`, searchlight patches will not use data outside of this mask.
        Defaults to ``None``, in which case all voxels are included in the
        analysis.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Returns
    -------
    rsa_results : 3D Nifti1Image | list of 3D Nifti1Image
        The correlation values for each searchlight patch. When multiple models
        have been supplied, a list will be returned containing the RSA results
        for each model.

    See Also
    --------
    compute_dsm
    """
    # Check for compatibility of the source estimates and the model features
    one_model = type(dsm_model) is np.ndarray
    if one_model:
        dsm_model = [dsm_model]

    if (not isinstance(image, tuple(nib.imageclasses.all_image_classes))
            or image.ndim != 4):
        raise ValueError('The image data must be 4-dimensional Nifti-like '
                         'images')

    # Check for compatibility of the BOLD images and the model features
    for dsm in dsm_model:
        n_items = _n_items_from_dsm(dsm)
        if image.shape[3] != n_items and y is None:
            raise ValueError(
                'The number of images (%d) should be equal to the '
                'number of items in `dsm_model` (%d). Alternatively, use '
                'the `y` parameter to assign evokeds to items.'
                % (image.shape[3], n_items))
        if y is not None and len(np.unique(y)) != n_items:
            raise ValueError(
                'The number of items in `dsm_model` (%d) does not match '
                'the number of items encoded in the `y` matrix (%d).'
                % (n_items, len(np.unique(y))))

    # Get data as (n_items x n_voxels)
    X = image.get_fdata().reshape(-1, image.shape[3]).T

    # Find voxel positions
    voxels = np.array(list(np.ndindex(image.shape[:-1])))
    voxel_loc = voxels @ image.affine[:3, :3]
    voxel_loc /= 1000  # convert position from mm to meters

    # Apply masks
    result_mask = np.ones(image.shape[:3], dtype=bool)
    if brain_mask is not None:
        if brain_mask.ndim != 3 or brain_mask.shape != image.shape[:3]:
            raise ValueError('Brain mask must be a 3-dimensional Nifi-like '
                             'image with the same dimensions as the data '
                             'image')
        brain_mask = brain_mask.get_fdata() != 0
        result_mask &= brain_mask
        brain_mask = brain_mask.ravel()
        X = X[:, brain_mask]
        voxel_loc = voxel_loc[brain_mask]
    if roi_mask is not None:
        if roi_mask.ndim != 3 or roi_mask.shape != image.shape[:3]:
            raise ValueError('ROI mask must be a 3-dimensional Nifi-like '
                             'image with the same dimensions as the data '
                             'image')
        roi_mask = roi_mask.get_fdata() != 0
        result_mask &= roi_mask
        roi_mask = roi_mask.ravel()
        if brain_mask is not None:
            roi_mask = roi_mask[brain_mask]
        roi_mask = np.flatnonzero(roi_mask)

    # Compute distances between voxels
    logger.info('Computing distances...')
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(radius=spatial_radius, n_jobs=n_jobs).fit(voxel_loc)
    dist = nn.radius_neighbors_graph(mode='distance')

    # Perform the RSA
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=None, sel_series=roi_mask)
    rsa_result = rsa_array(X, dsm_model, patches,
                           data_dsm_metric=image_dsm_metric,
                           data_dsm_params=image_dsm_params,
                           rsa_metric=rsa_metric, ignore_nan=ignore_nan, y=y,
                           n_folds=n_folds, n_jobs=n_jobs, verbose=verbose)

    if one_model:
        data = np.zeros(image.shape[:3])
        data[result_mask] = rsa_result
        return nib.Nifti1Image(data, image.affine, image.header)
    else:
        results = []
        for i in range(rsa_result.shape[-1]):
            data = np.zeros(image.shape[:3])
            data[result_mask] = rsa_result[:, i]
            results.append(nib.Nifti1Image(data, image.affine, image.header))
        return results


def dsm_nifti(image, spatial_radius=0.01, dist_metric='correlation',
              dist_params=dict(), y=None, n_folds=1, roi_mask=None,
              brain_mask=None, n_jobs=1, verbose=False):
    """Generate DSMs in a searchlight pattern on Nibabel Nifty-like images.

    DSMs are computed using a patch surrounding each voxel.

    .. versionadded:: 0.4

    Parameters
    ----------
    image : 4D Nifti-like image
        The Nitfi image data. The 4th dimension must contain the images
        for each item.
    spatial_radius : float
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch.
        Defaults to 0.01.
    dist_metric : str
        The metric to use to compute the DSM for the data. This can be
        any metric supported by the scipy.distance.pdist function. See also the
        ``dist_params`` parameter to specify and additional parameter for
        the distance function. Defaults to 'correlation'.
    dist_params : dict
        Extra arguments for the distance metric used to compute the DSMs.
        Refer to :mod:`scipy.spatial.distance` for a list of all other metrics
        and their arguments. Defaults to an empty dictionary.
    y : ndarray of int, shape (n_items,) | None
        For each source estimate, a number indicating the item to which it
        belongs. When ``None``, each source estimate is assumed to belong to a
        different item. Defaults to ``None``.
    n_folds : int | sklearn.model_selection.BaseCrollValidator | None
        Number of cross-validation folds to use when computing the distance
        metric. Folds are created based on the ``y`` parameter. Specify
        ``None`` to use the maximum number of folds possible, given the data.
        Alternatively, you can pass a Scikit-Learn cross validator object (e.g.
        ``sklearn.model_selection.KFold``) to assert fine-grained control over
        how folds are created.
        Defaults to 1 (no cross-validation).
    roi_mask : 3D Nifti-like image | None
        When set, searchlight patches will only be generated for the subset of
        voxels with non-zero values in the given mask. This is useful for
        restricting the analysis to a region of interest (ROI). Note that while
        the center of the patches are all within the ROI, the patch itself may
        extend beyond the ROI boundaries.
        Defaults to ``None``, in which case patches for all voxels are
        generated.
    brain_mask : 3D Nifti-like image | None
        When set, searchlight patches are restricted to only contain voxels
        with non-zero values in the given mask. This is useful for make sure
        only information from inside the brain is used. In contrast to the
        `roi_mask`, searchlight patches will not use data outside of this mask.
        Defaults to ``None``, in which case all voxels are included in the
        analysis.
    n_jobs : int
        The number of processes (=number of CPU cores) to use. Specify -1 to
        use all available cores. Defaults to 1.
    verbose : bool
        Whether to display a progress bar. In order for this to work, you need
        the tqdm python module installed. Defaults to False.

    Yields
    ------
    dsm : ndarray, shape (n_items, n_items)
        A DSM for each searchlight patch.
    """
    if (not isinstance(image, tuple(nib.imageclasses.all_image_classes))
            or image.ndim != 4):
        raise ValueError('The image data must be 4-dimensional Nifti-like '
                         'images')

    # Get data as (n_items x n_voxels)
    X = image.get_fdata().reshape(-1, image.shape[3]).T

    # Find voxel positions
    voxels = np.array(list(np.ndindex(image.shape[:-1])))
    voxel_loc = voxels @ image.affine[:3, :3]
    voxel_loc /= 1000  # convert position from mm to meters

    # Apply masks
    result_mask = np.ones(image.shape[:3], dtype=bool)
    if brain_mask is not None:
        if brain_mask.ndim != 3 or brain_mask.shape != image.shape[:3]:
            raise ValueError('Brain mask must be a 3-dimensional Nifi-like '
                             'image with the same dimensions as the data '
                             'image')
        brain_mask = brain_mask.get_fdata() != 0
        result_mask &= brain_mask
        brain_mask = brain_mask.ravel()
        X = X[:, brain_mask]
        voxel_loc = voxel_loc[brain_mask]
    if roi_mask is not None:
        if roi_mask.ndim != 3 or roi_mask.shape != image.shape[:3]:
            raise ValueError('ROI mask must be a 3-dimensional Nifi-like '
                             'image with the same dimensions as the data '
                             'image')
        roi_mask = roi_mask.get_fdata() != 0
        result_mask &= roi_mask
        roi_mask = roi_mask.ravel()
        if brain_mask is not None:
            roi_mask = roi_mask[brain_mask]
        roi_mask = np.flatnonzero(roi_mask)

    # Compute distances between voxels
    logger.info('Computing distances...')
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(radius=spatial_radius, n_jobs=n_jobs).fit(voxel_loc)
    dist = nn.radius_neighbors_graph(mode='distance')

    # Compute DSMs
    patches = searchlight(X.shape, dist=dist, spatial_radius=spatial_radius,
                          temporal_radius=None, sel_series=roi_mask)
    yield from dsm_array(X, patches, dist_metric=dist_metric,
                         dist_params=dist_params, y=y, n_folds=n_folds,
                         n_jobs=n_jobs, verbose=verbose)


def _check_stcs_compatibility(stcs, src):
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

    Code is mostly taken from `mne.add_source_space_distances`.

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
    rows, cols = np.nonzero(neighbors != -1)
    cols = neighbors[(rows, cols)]
    dist = np.linalg.norm(src[0]['rr'][rows, :] - src[0]['rr'][cols, :],
                          axis=1)
    con_matrix = csr_matrix((dist, (rows, cols)), shape=(n_sources, n_sources))
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
    src[0]['dist_limit'] = np.array([dist_limit], 'float32')
    return src


def make_mri_con_matrix(img):
    from scipy.sparse import csr_matrix
    # Create 3 x 3 x 3 cube of (ijk) indices, centered around (0, 0, 0)
    cube = np.array(list(np.ndindex(3, 3, 3))) - [1, 1, 1]
    # Remove center of the cube
    cube = np.delete(cube, len(cube) // 2, axis=0)
    # Compute distance from all points in the cube to the center
    dist = np.linalg.norm(cube @ img.affine[:3, :3], axis=1)
    # Copy the cube, centering it around each voxel
    voxels = np.array(list(np.ndindex(*img.shape[:3])))
    neighbours = voxels[:, :, np.newaxis] + cube.T[np.newaxis, :, :]
    assert neighbours.shape == (len(voxels), 3, len(cube))
    # Convert ijk coordinates to voxel numbers
    neighbours = np.ravel_multi_index(
        (neighbours[:, 0, :], neighbours[:, 1, :], neighbours[:, 2, :]),
        img.shape[:3],
        mode='clip',
    )
    rows = np.repeat(np.arange(neighbours.shape[0]), neighbours.shape[1])
    cols = neighbours.ravel()
    dist = np.tile(dist, neighbours.shape[0])
    con_matrix = csr_matrix((dist, (rows, cols)),
                            shape=(len(voxels), len(voxels)))
    return con_matrix


def backfill_stc_from_rois(values, rois, src, tmin=0, tstep=1, subject=None):
    """Backfill the ROI values into a full mne.SourceEstimate object.

    Each vertex belonging to the same region of interest (ROI) will have the
    sample value.

    Parameters
    ----------
    values : ndarray, shape (n_rois, ...)
        For each ROI, either a single value or a timecourse of values.
    rois : list of mne.Label
        The spatial regions of interest (ROIs) to compute the RSA for. This
        needs to be specified as a list of ``mne.Label`` objects, such as
        returned by ``mne.read_annotations``.
    src : instance of mne.SourceSpaces
        The source space used by the source estimates specified in the `stcs`
        parameter.
    tmin : float
        Time corrsponding to the first sample.
    tstep : float
        Difference in time between two samples.
    subject : str | None
        The name of the FreeSurfer subject.

    Returns
    -------
    stc : mne.SourceEstimate
        The backfilled source estimate object.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        n_samples = 1
    else:
        n_samples = values.shape[1]
    data = np.zeros((src[0]['nuse'] + src[1]['nuse'], n_samples))
    verts_lh = src[0]['vertno']
    verts_rh = src[1]['vertno']
    for roi, rsa_timecourse in zip(rois, values):
        roi = roi.copy().restrict(src)
        if roi.hemi == 'lh':
            roi_ind = np.searchsorted(verts_lh, roi.vertices)
        else:
            roi_ind = np.searchsorted(verts_rh, roi.vertices)
            roi_ind += src[0]['nuse']
        for ind in roi_ind:
            data[ind] = rsa_timecourse
    return mne.SourceEstimate(data, vertices=[verts_lh, verts_rh], tmin=tmin,
                              tstep=tstep, subject=subject)
