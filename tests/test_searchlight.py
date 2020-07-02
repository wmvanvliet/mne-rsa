import pytest
import numpy as np
from numpy.testing import assert_equal
from mne_rsa import searchlight


class TestSearchLight:
    """Test the searchlight generator class"""

    def test_iter_spatio_temporal(self):
        """Test generating spatio-temporal searchlight patches."""
        dist = np.array([[0, 1, 2],
                         [1, 0, 1],
                         [2, 1, 0]])
        s = searchlight((10, 3, 4), dist, spatial_radius=2, temporal_radius=1)
        assert len(s) == 6
        assert s.shape == (3, 2)
        assert_equal(list(s), [
            (slice(None, None), np.array([0, 1]), slice(0, 3)),
            (slice(None, None), np.array([0, 1]), slice(1, 4)),
            (slice(None, None), np.array([0, 1, 2]), slice(0, 3)),
            (slice(None, None), np.array([0, 1, 2]), slice(1, 4)),
            (slice(None, None), np.array([1, 2]), slice(0, 3)),
            (slice(None, None), np.array([1, 2]), slice(1, 4)),
        ])

    def test_iter_spatial(self):
        """Test generating spatial searchlight patches."""
        slice_all = slice(None, None)  # To save space
        dist = np.array([[0, 1, 2],
                         [1, 0, 1],
                         [2, 1, 0]])

        s = searchlight((5, 10, 3, 4), dist, spatial_radius=2)
        assert len(s) == 3
        assert s.shape == (3,)
        assert_equal(list(s), [
            (slice_all, slice_all, np.array([0, 1]), slice_all),
            (slice_all, slice_all, np.array([0, 1, 2]), slice_all),
            (slice_all, slice_all, np.array([1, 2]), slice_all),
        ])

        s = searchlight((10, 3, 4), dist, spatial_radius=2)
        assert len(s) == 3
        assert s.shape == (3,)
        assert_equal(list(s), [
            (slice_all, np.array([0, 1]), slice_all),
            (slice_all, np.array([0, 1, 2]), slice_all),
            (slice_all, np.array([1, 2]), slice_all),
        ])

        s = searchlight((10, 3), dist, spatial_radius=2)
        assert len(s) == 3
        assert s.shape == (3,)
        assert_equal(list(s), [
            (slice_all, np.array([0, 1])),
            (slice_all, np.array([0, 1, 2])),
            (slice_all, np.array([1, 2])),
        ])

    def test_iter_temporal(self):
        """Test generating temporal searchlight patches."""
        s = searchlight((10, 3, 4), temporal_radius=1)
        assert len(s) == 2
        assert s.shape == (2,)
        assert_equal(list(s), [
            (slice(None, None), slice(None, None), slice(0, 3)),
            (slice(None, None), slice(None, None), slice(1, 4)),
        ])

        s = searchlight((10, 4), temporal_radius=1)
        assert len(s) == 2
        assert s.shape == (2,)
        assert_equal(list(s), [
            (slice(None, None), slice(0, 3)),
            (slice(None, None), slice(1, 4)),
        ])

    def test_iter_single(self):
        """Test generating a single searchlight patch."""
        s = searchlight((5, 10, 3, 4))
        assert len(s) == 1
        assert s.shape == tuple()
        assert_equal(next(iter(s)), tuple([slice(None, None)] * 4))

        s = searchlight((10, 3, 4))
        assert len(s) == 1
        assert s.shape == tuple()
        assert_equal(next(iter(s)), tuple([slice(None, None)] * 3))

        s = searchlight((10, 3))
        assert len(s) == 1
        assert s.shape == tuple()
        assert_equal(next(iter(s)), tuple([slice(None, None)] * 2))

        s = searchlight((10,))
        assert len(s) == 1
        assert s.shape == tuple()
        assert_equal(next(iter(s)), (slice(None, None),))

    def test_invalid_input(self):
        """Test giving invalid input to searchlight generator."""
        dist = np.array([[0, 1, 2],
                         [1, 0, 1],
                         [2, 1, 0]])

        # Spatio-temporal case
        with pytest.raises(ValueError, match='no spatial dimension'):
            searchlight((10,), dist=dist, spatial_radius=1, temporal_radius=1)
        with pytest.raises(ValueError, match='no temporal dimension'):
            searchlight((10, 3), dist=dist,
                        spatial_radius=1, temporal_radius=1)
        with pytest.raises(ValueError, match='no distance information'):
            searchlight((10, 3, 4), spatial_radius=1, temporal_radius=1)

        # Spatial case
        with pytest.raises(ValueError, match='no spatial dimension'):
            searchlight((10,), dist=dist, spatial_radius=1)
        with pytest.raises(ValueError, match='no distance information'):
            searchlight((10, 5), spatial_radius=1)

        # Temporal case
        with pytest.raises(ValueError, match='no temporal dimension'):
            searchlight((10,), dist=dist, temporal_radius=1)
