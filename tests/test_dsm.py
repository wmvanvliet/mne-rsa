import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose

from mne_rsa import searchlight, dsm_array, compute_dsm, compute_dsm_cv


class TestDsm:
    """Test computing a DSM"""

    def test_basic(self):
        """Test basic invocation of compute_dsm."""
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        dsm = compute_dsm(data)
        assert dsm.shape == (1,)
        assert_allclose(dsm, 0, atol=1E-15)

    def test_invalid_input(self):
        """Test giving invalid input to compute_dsm."""
        data = np.array([[1], [1]])
        with pytest.raises(ValueError, match='single feature'):
            compute_dsm(data, metric='correlation')

    def test_set_metric(self):
        """Test setting distance metric for computing DSMs."""
        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]])
        dsm = compute_dsm(data, metric='euclidean')
        assert dsm.shape == (1,)
        assert_allclose(dsm, 5.477226)


class TestDsmCV:
    """Test computing a DSM with cross-validation"""

    def test_basic(self):
        """Test basic invocation of compute_dsm."""
        data = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],
                         [[1, 2, 3, 4], [1, 2, 3, 4]]])
        dsm = compute_dsm_cv(data)
        assert dsm.shape == (1,)
        assert_allclose(dsm, 0, atol=1E-15)

    def test_invalid_input(self):
        """Test giving invalid input to compute_dsm."""
        data = np.array([[[1], [1]]])
        with pytest.raises(ValueError, match='single feature'):
            compute_dsm_cv(data, metric='correlation')

    def test_set_metric(self):
        """Test setting distance metric for computing DSMs."""
        data = np.array([[[1, 2, 3, 4], [2, 4, 6, 8]],
                         [[1, 2, 3, 4], [2, 4, 6, 8]]])
        dsm = compute_dsm_cv(data, metric='euclidean')
        assert dsm.shape == (1,)
        assert_allclose(dsm, 5.477226)


class TestDsmsSearchlight:
    """Test computing DSMs with searchlight patches."""

    def test_temporal(self):
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        patches = searchlight(data.shape, temporal_radius=1)
        dsms = dsm_array(data, patches, dist_metric='euclidean')
        assert len(dsms) == 2
        assert dsms.shape == (2, 1)
        assert_equal(list(dsms), [0, 0])