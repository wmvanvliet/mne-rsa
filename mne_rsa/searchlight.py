import numpy as np
from mne.utils import logger


class searchlight:
    """Generate indices for searchlight patches.

    Generates a sequence of tuples that can be used to index a data array.
    Depending on the spatial and temporal radius, each tuple extracts a
    searchlight patch along time, space or both.

    This function is flexible in regards to shape of the data array. The
    intepretation of the dimensions is as follows::

    4 or more dimensions
        ``(n_folds, n_items, n_series, n_samples, ...)``

    3 dimensions
        ``(n_items, n_series, n_samples)``

    2 dimensions
        ``(n_items, n_series)`` when ``spatial_radius`` is not ``None``.
        ``(n_items, n_samples)`` when ``temporal_radius`` is not ``None``.

    1 dimension
        ``(n_items,)``

    The returned tuples will match the dimensions of the data array.

    Parameters
    ----------
    shape : tuple of int
        The shape of the data array to compute the searchlight patches for, as
        obtained with the ``.shape`` attribute.
    dist : ndarray or sparse matrix, shape (n_series, n_series) | None
        The distances between all source points or sensors in meters.
        This parameter needs to be specified if a ``spatial_radius`` is set.

        Since the distance matrix can be huge, sparse matrices are also
        supported. When the distance matrix is sparse, all zero distances are
        treated as infinity. This allows you to skip far away points during
        your distance computations.

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

    Attributes
    ----------
    shape
    """
    def __init__(self, shape, dist=None, spatial_radius=None,
                 temporal_radius=None, sel_series=None,
                 sel_samples=None):
        # Interpret the dimensions of the data array (see docstring)
        n_dims = len(shape)
        if n_dims >= 4:
            self.series_dim = 2
            self.samples_dim = 3
        elif n_dims == 3:
            self.series_dim = 1
            self.samples_dim = 2
        elif n_dims == 2 and spatial_radius is not None:
            self.series_dim = 1
            self.samples_dim = None
        elif n_dims == 2 and temporal_radius is not None:
            self.series_dim = None
            self.samples_dim = 1
        else:
            self.series_dim = None
            self.samples_dim = None

        self.dist = dist
        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.sel_series = sel_series
        self.sel_samples = sel_samples

        # Will we be creating spatial searchlight patches?
        if self.spatial_radius is not None:
            if self.dist is None:
                raise ValueError('A spatial radius was requested, but no '
                                 'distance information was specified '
                                 '(=dist parameter).')
            if self.series_dim is None:
                raise ValueError('Cannot create spatial searchlight patches: '
                                 f'the provided data matrix {shape} has no '
                                 'spatial dimension.')
            if self.sel_series is None:
                self.sel_series = np.arange(shape[self.series_dim])

            # Compressed Sparse Row format is optimal for our computations
            from scipy.sparse import issparse
            if issparse(self.dist):
                self.dist = self.dist.tocsr()
        else:
            self.sel_series = None

        # Will we be creating temporal searchlight patches?
        if temporal_radius is not None:
            if self.samples_dim is None:
                raise ValueError('Cannot create temporal searchlight patches: '
                                 f'the provided data matrix {shape} has no '
                                 'temporal dimension.')
            n_samples = shape[self.samples_dim]
            if self.sel_samples is None:
                self.sel_samples = np.arange(n_samples)
            # Get a valid time range given the size of the sliding windows
            self.time_centers = [
                t for t in self.sel_samples
                if t - temporal_radius >= 0 and t + temporal_radius < n_samples
            ]
        else:
            self.sel_samples = None

        # Create a template for the patches that will be generated that is
        # compatible with the data array dimensions. By default, we select
        # everything along every dimension. This template will be filled-in
        # inside the __iter__ function.
        self.patch_template = [slice(None)] * n_dims

        # Setup the main generator function that will be providing the
        # searchlight patches.
        if (self.spatial_radius is not None
                and self.temporal_radius is not None):
            self._generator = self._iter_spatio_temporal()
        elif self.spatial_radius is not None:
            self._generator = self._iter_spatial()
        elif self.temporal_radius is not None:
            self._generator = self._iter_temporal()
        else:
            # Single searchlight patch only
            self._generator = iter([tuple(self.patch_template)])

    def __iter__(self):
        return self

    def __next__(self):
        """Generate searchlight patches."""
        return next(self._generator)

    def _iter_spatio_temporal(self):
        """Generate spatio-temporal searchlight patches."""
        logger.info('Creating spatio-temporal searchlight patches')
        patch = list(self.patch_template)  # Copy the template
        for series in self.sel_series:
            # Compute all spatial locations in the searchligh path.
            spat_ind = _get_in_radius(self.dist, series, self.spatial_radius)
            patch[self.series_dim] = spat_ind
            for sample in self.time_centers:
                temp_ind = slice(sample - self.temporal_radius,
                                 sample + self.temporal_radius + 1)
                patch[self.samples_dim] = temp_ind
                yield tuple(patch)

    def _iter_spatial(self):
        """Generate spatial searchlight patches only."""
        logger.info('Creating spatial searchlight patches')
        patch = list(self.patch_template)  # Copy the template
        for series in self.sel_series:
            spat_ind = _get_in_radius(self.dist, series, self.spatial_radius)
            patch[self.series_dim] = spat_ind
            yield tuple(patch)

    def _iter_temporal(self):
        """Generate temporal searchlight patches only."""
        logger.info('Creating temporal searchlight patches')
        patch = list(self.patch_template)  # Copy the template
        for sample in self.time_centers:
            patch[self.samples_dim] = slice(sample - self.temporal_radius,
                                            sample + self.temporal_radius + 1)
            yield tuple(patch)

    @property
    def shape(self):
        """Number of generated patches along multiple dimensions.

        This is useful for re-shaping the result obtained after consuming the
        this generator.

        Returns
        -------
        shape : tuple of int
            For a spatio-temporal searchlight:
                Two elements: the number of time-series and number of time
                samples for which patches are generated.
            For a spatial searchlight:
                One element: the number of time-series for which patches are
                generated.
            For a temporal searchlight:
                One element: the number of time-samples for which patches are
                generated.
            For no searchlight:
                Zero elements.
        """
        if (self.spatial_radius is not None
                and self.temporal_radius is not None):
            return (len(self.sel_series), len(self.time_centers))
        elif self.spatial_radius is not None:
            return (len(self.sel_series),)
        elif self.temporal_radius is not None:
            return (len(self.time_centers),)
        else:
            return tuple()

    def __len__(self):
        """Get total number of searchlight patches that will be generated."""
        total = 1
        for n in self.shape:  # Number of patches generated in each dimension
            total *= n
        return total


def _get_in_radius(dist, seed, radius):
    """Obtain indices for all points within the given radius from a seed point.

    Takes care to work with sparse matrices too.

    Parameters
    ----------
    dist : ndarray or sparse matrix, shape (n_points, n_points)
        The distances between all points.
    seed : int
        The index of the point used as a seed.
    radius : float
        The maximum distance that points can be to be included.

    Returns
    -------
    ind : ndarray, shape (n_points_in_radius,)
        Indices of all points in the given radius from the seed point.
    """
    from scipy.sparse import issparse
    if issparse(dist):
        # Treat all zero distances as missing data
        ind = dist[seed].nonzero()[1]
        # Find indices for points within the radius
        ind = ind[dist[seed].data < radius]
        ind.sort()
        # Be sure to add the seed point, which has distance of 0 to itself
        ind = np.hstack((ind, [seed]))  # Di
    else:
        ind = np.flatnonzero(dist[seed] < radius)
    return sorted(ind)
