import numpy as np


def searchlight_patches(shape, dist=None, spatial_radius=None,
                        temporal_radius=None, sel_series=None,
                        sel_samples=None):
    """Generate indices for searchlight patches.

    Generates a sequence of tuples that can be used to index a data array of
    shape ``(n_series, n_samples)``. Depending on the spatial and temporal
    radius, each tuple extracts a searchlight patch along time, space or both.

    This function is flexible in regards to shape of the data array. The
    intepretation of the dimensions is as follows::

    4 dimensions
        ``(n_folds, n_items, n_series, n_samples)``

    3 dimensions
        ``(n_items, n_series, n_samples)``

    2 dimensions
        ``(n_items, n_series)`` when ``spatial_radius`` is not ``None``.
        ``(n_items, n_samples)`` when ``temporal_radius`` is not ``None``.

    1 dimension
        ``(n_items,)``

    Parameters
    ----------
    shape : tuple of int
        The shape of the data array to compute the searchlight patches for, as
        obtained with the ``.shape`` attribute.
    dist : ndarray, shape (n_series, n_series) | None
        The distances between all source points or sensors in meters.
        This parameter needs to be specified if a ``spatial_radius`` is set.
        Defaults to ``None``.
    spatial_radius : floats | None
        The spatial radius of the searchlight patch in meters. All source
        points within this radius will belong to the searchlight patch. Set to
        None to only perform the searchlight over time. When this parameter is
        set, the ``dist`` parameter must also be specified. Defaults to
        ``None``.
    temporal_radius : float | None
        The temporal radius of the searchlight patch in samples. Set to
        ``None`` to only perform the searchlight over sensors/source points.
        Defaults to ``None``.
    sel_series : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time series with the given indices. Defaults to ``None``, in which case
        patches for all series are generated.
    sel_samples : ndarray, shape (n_selected_series,) | None
        When set, searchlight patches will only be generated for the subset of
        time samples with the given indices. Defaults to ``None``, in which
        case patches for all samples are generated.

    Yields
    ------
    patch : tuple of (slice | ndarray)
        A single searchlight patch. Each element of the tuple corresponds to a
        dimension of the data array and can be used to index along this
        dimension to extract the searchlight patch.
    """
    # Interpret the dimensions of the data array (see docstring for details)
    n_dims = len(shape)
    if n_dims >= 4:
        series_dim = 2
        samples_dim = 3
    elif n_dims == 3:
        series_dim = 1
        samples_dim = 2
    elif n_dims == 2 and spatial_radius is not None:
        series_dim = 1
        samples_dim = None
    elif n_dims == 2 and temporal_radius is not None:
        series_dim = None
        samples_dim = 1
    else:
        series_dim = None
        samples_dim = None

    n_series = shape[series_dim] if series_dim is not None else None
    n_samples = shape[samples_dim] if samples_dim is not None else None

    # Will we be creating spatial searchlight patches?
    if spatial_radius is not None:
        if dist is None:
            raise ValueError('A spatial radius was requested, but no distance '
                             'information was specified (=dist parameter).')
        if n_series is None:
            raise ValueError('Cannot create spatial searchlight patches: the '
                             f'provided data matrix {shape} has no spatial '
                             'dimension.')
        if sel_series is None:
            sel_series = np.arange(n_series)

    # Will we be creating temporal searchlight patches?
    if temporal_radius is not None:
        if n_samples is None:
            raise ValueError('Cannot create temporal searchlight patches: the '
                             f'provided data matrix {shape} has no temporal '
                             'dimension.')
        if sel_samples is None:
            sel_samples = np.arange(n_samples)
        # Get a valid time range given the size of the sliding windows
        time_centers = [
            t for t in sel_samples
            if t - temporal_radius >= 0 and t + temporal_radius < n_samples
        ]

    # Create a template for the patches that will be generated that is
    # compatible with the data array dimensions. By default, we select
    # everything along every dimension. This template will be filled-in below.
    patch_template = [slice(None, None)] * n_dims

    # Spatio-temporal searchlight
    if spatial_radius is not None and temporal_radius is not None:
        for series in sel_series:
            for sample in time_centers:
                patch = list(patch_template)  # Copy the template
                patch[series_dim], patch[samples_dim] = (
                    np.flatnonzero(dist[series] < spatial_radius),
                    slice(sample - temporal_radius,
                          sample + temporal_radius + 1))
                yield tuple(patch)

    # Spatial searchlight only
    elif spatial_radius is not None:
        for series in sel_series:
            patch = list(patch_template)  # Copy the template
            patch[series_dim] = np.flatnonzero(dist[series] < spatial_radius)
            yield tuple(patch)

    # Temporal searchlight only
    elif temporal_radius is not None:
        for sample in time_centers:
            patch = list(patch_template)  # Copy the template
            patch[samples_dim] = slice(sample - temporal_radius,
                                       sample + temporal_radius + 1)
            yield tuple(patch)

    # Only a single searchlight patch
    else:
        yield tuple(patch_template)
