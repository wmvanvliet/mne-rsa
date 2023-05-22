import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from mne_rsa import searchlight, dsm_array, compute_dsm, compute_dsm_cv
from mne_rsa.dsm import _ensure_condensed, _n_items_from_dsm


class TestDsm:
    """Test computing a DSM"""

    def test_basic(self):
        """Test basic invocation of compute_dsm."""
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        dsm = compute_dsm(data)
        assert dsm.shape == (1,)
        assert_allclose(dsm, 0, atol=1e-15)

    def test_invalid_input(self):
        """Test giving invalid input to compute_dsm."""
        data = np.array([[1], [1]])
        with pytest.raises(ValueError, match="single feature"):
            compute_dsm(data, metric="correlation")

    def test_set_metric(self):
        """Test setting distance metric for computing DSMs."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]])
        dsm = compute_dsm(data, metric="euclidean")
        assert dsm.shape == (1,)
        assert_allclose(dsm, 5.477226)


class TestDsmCV:
    """Test computing a DSM with cross-validation."""

    def test_basic(self):
        """Test basic invocation of compute_dsm_cv."""
        data = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        dsm = compute_dsm_cv(data)
        assert dsm.shape == (1,)
        assert_allclose(dsm, 0, atol=1e-15)

    def test_invalid_input(self):
        """Test giving invalid input to compute_dsm."""
        data = np.array([[[1], [1]]])
        with pytest.raises(ValueError, match="single feature"):
            compute_dsm_cv(data, metric="correlation")

    def test_set_metric(self):
        """Test setting distance metric for computing DSMs."""
        data = np.array([[[1, 2, 3, 4], [2, 4, 6, 8]], [[1, 2, 3, 4], [2, 4, 6, 8]]])
        dsm = compute_dsm_cv(data, metric="euclidean")
        assert dsm.shape == (1,)
        assert_allclose(dsm, 5.477226)


class TestEnsureCondensed:
    """Test the _ensure_condensed function."""

    def test_basic(self):
        """Test basic invocation of _ensure_condensed."""
        dsm = _ensure_condensed(
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]), var_name="test"
        )
        assert dsm.shape == (3,)
        assert_equal(dsm, [1, 2, 3])

    def test_list(self):
        """Test invocation of _ensure_condensed on a list."""
        full = [
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
        ]
        dsm = _ensure_condensed(full, var_name="full")
        assert len(dsm) == 2
        assert dsm[0].shape == (3,)
        assert dsm[1].shape == (3,)
        assert_equal(dsm, [[1, 2, 3], [1, 2, 3]])

    def test_condensed(self):
        """Test invocation of _ensure_condensed on already condensed DSM."""
        dsm = _ensure_condensed(np.array([1, 2, 3]), var_name="test")
        assert dsm.shape == (3,)
        assert_equal(dsm, [1, 2, 3])

    def test_invalid(self):
        """Test _ensure_condensed with invalid inputs."""
        # Not a square matrix
        with pytest.raises(ValueError, match="square matrix"):
            _ensure_condensed(np.array([[0, 1], [1, 0], [2, 3]]), var_name="test")

        # Too many dimensions
        with pytest.raises(ValueError, match="Invalid dimensions"):
            _ensure_condensed(np.array([[[[[0, 1, 2, 3]]]]]), var_name="test")

        # Invalid type
        with pytest.raises(TypeError, match="NumPy array"):
            _ensure_condensed([1, 2, 3], var_name="test")


class TestNItemsFromDSM:
    """Test the _n_items_from_dsm function."""

    def test_basic(self):
        """Test basic invocation of _n_items_from_dsm."""
        assert _n_items_from_dsm(np.array([1, 2, 3])) == 3
        assert _n_items_from_dsm(np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])) == 3


class TestDsmsSearchlight:
    """Test computing DSMs with searchlight patches."""

    def test_temporal(self):
        """Test computing DSMs using a temporal searchlight."""
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        patches = searchlight(data.shape, temporal_radius=1)
        dsms = dsm_array(data, patches, dist_metric="euclidean")
        assert len(dsms) == len(patches)
        assert dsms.shape == (2, 1)
        assert_equal(list(dsms), [0, 0])

    def test_spatial(self):
        """Test computing DSMs using a spatial searchlight."""
        dist = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        patches = searchlight(data.shape, dist, spatial_radius=1)
        dsms = dsm_array(data, patches, dist_metric="euclidean")
        assert len(dsms) == len(patches)
        assert dsms.shape == (4, 1)
        assert_equal(list(dsms), [0, 0, 0, 0])

    def test_spatio_temporal(self):
        """Test computing DSMs using a spatio-temporal searchlight."""
        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        dsms = dsm_array(data, patches, dist_metric="correlation")
        assert len(dsms) == len(patches)
        assert dsms.shape == (2, 1, 3)
        assert_allclose(list(dsms), [[0, 0, 0], [0, 0, 0]], atol=1e-15)

    def test_single_patch(self):
        """Test computing DSMs using a single searchlight patch."""
        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        dsms = dsm_array(data, dist_metric="correlation")
        assert len(dsms) == 1
        assert dsms.shape == (3,)
        assert_allclose(list(dsms), [[0, 0, 0]], atol=1e-15)

    def test_crossvalidation(self):
        """Test computing DSMs using a searchlight and cross-validation."""
        data = np.array(
            [
                [[1, 2, 3], [2, 3, 4]],
                [[2, 3, 4], [3, 4, 5]],
                [[3, 4, 5], [4, 5, 6]],
                [[1, 2, 3], [2, 3, 4]],
                [[2, 3, 4], [3, 4, 5]],
                [[3, 4, 5], [4, 5, 6]],
            ]
        )
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        dsms = dsm_array(data, patches, y=[1, 2, 3, 1, 2, 3], n_folds=2)
        assert len(dsms) == len(patches)
        assert dsms.shape == (2, 1, 3)
        assert_allclose(list(dsms), [[0, 0, 0], [0, 0, 0]], atol=1e-15)
