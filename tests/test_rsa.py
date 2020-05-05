import pytest
from types import GeneratorType
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from mne_rsa import rsa, rsa_gen, rsa_array


def dsm():
    """Create an example DSM"""
    return np.array([1, 2, 3, 4, 5, 6])


def dsm_gen(dsms):
    """Generator for DSMs"""
    for dsm in dsms:
        yield np.asarray(dsm)


class TestRSAGen:
    """Test the rsa_gen function"""
    def test_return_type(self):
        """Test return type of rsa_gen"""
        assert isinstance(rsa_gen(dsm_gen([dsm()]), dsm()), GeneratorType)
        assert next(rsa_gen(dsm_gen([dsm()]), dsm())).shape == tuple()
        assert next(rsa_gen(dsm_gen([dsm()]), [dsm()])).shape == (1,)

    def test_spearman(self):
        """Test computing RSA with Spearman correlation"""
        data_dsm = dsm_gen([[1, 2, 3]])
        model_dsm = np.array([2, 3, 3.5])
        assert next(rsa_gen(data_dsm, model_dsm, metric='spearman')) == 1.0

    def test_pearson(self):
        """Test computing RSA with Pearson correlation"""
        data_dsm = dsm_gen([[1, 2, 3]])
        model_dsm = np.array([2, 3, 3.5])
        assert next(rsa_gen(data_dsm, model_dsm, metric='pearson')) < 1.0

    def test_kendall_tau_a(self):
        """Test computing RSA with Kendall's Tau Alpha"""
        data_dsm = dsm_gen([[1, 2, 3]])
        model_dsm = np.array([1, 3, 3])  # This metric deals well with ties
        rsa_val = next(rsa_gen(data_dsm, model_dsm, metric='kendall-tau-a'))
        assert rsa_val == 2 / 3

    def test_regression(self):
        """Test computing RSA with regression."""
        model_dsm1 = np.array([-1, 0, 1])
        model_dsm2 = np.array([1, -2, 1])
        data_dsm = dsm_gen([3 * model_dsm1 + 5 * model_dsm2])
        rsa_val = next(rsa_gen(data_dsm, [model_dsm1, model_dsm2],
                               metric='regression'))
        assert_allclose(rsa_val, [3, 5])

    def test_partial(self):
        """Test computing RSA with partial correlation."""
        # Example taken from https://en.wikipedia.org/wiki/Partial_correlation
        model_dsm1 = np.array([1, 2, 3, 4])
        model_dsm2 = np.array([0, 0, 1, 1])
        data_dsm = dsm_gen([[2, 4, 15, 20]])
        rsa_val = next(rsa_gen(data_dsm, [model_dsm1, model_dsm2],
                               metric='partial'))
        assert_allclose(rsa_val, [0.919145, 0.912871])

    def test_partial_spearman(self):
        """Test computing RSA with partial spearman correlation."""
        # Example verified with MATLAB's partialcorr function
        model_dsm1 = np.array([1, 2, 3, 4])
        model_dsm2 = np.array([0, 0, 1, 1])
        data_dsm = dsm_gen([[2, 4, 20, 15]])
        rsa_val = next(rsa_gen(data_dsm, [model_dsm1, model_dsm2],
                               metric='partial-spearman'))
        assert_allclose(rsa_val, [0, 2 / 3], atol=1E-15)

    def test_invalid_metric(self):
        """Test whether computing RSA with an invalid metric raises error."""
        with pytest.raises(ValueError, match='Invalid RSA metric'):
            next(rsa_gen(dsm_gen([dsm()]), dsm(), metric='foo'))


class TestRSA:
    """Test the main RSA function"""
    # Most of the functionality is already tested in TestRSAGen
    def test_return_type(self):
        """Test return type of rsa_gen"""
        assert isinstance(rsa([dsm()], dsm()), np.ndarray)
        assert rsa(dsm(), dsm()).shape == tuple()
        assert rsa(dsm(), [dsm()]).shape == (1,)


class TestRSAArray:
    """Test computing RSA on a NumPy array"""
    def invalid_input(self):
        """Test invalid inputs."""
        data = np.array([[1], [2], [3], [4]])
        model_dsm = dsm()
        with pytest.raises(ValueError, match='There is only a single feature'):
            rsa_array(data, model_dsm)

    def test_rsa_no_searchlight(self):
        """Test RSA without searchlight patches."""
        data = np.array([[1], [2], [3], [4]])
        model_dsm = np.array([1, 2, 3, 1, 2, 1])
        rsa_result = rsa_array(data, model_dsm, data_dsm_metric='euclidean')
        assert rsa_result.shape == (1, 1, 1)
        assert rsa_result == 1

    def test_rsa_temp(self):
        """Test RSA with a temporal searchlight."""
        data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        model_dsm = np.array([1, 2, 1])

        # One model DSM, no paralellization
        rsa_result = rsa_array(data, model_dsm,
                               temporal_radius=1, spatial_radius=None,
                               data_dsm_metric='euclidean')
        assert rsa_result.shape == (1, 2)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, no paralellization
        rsa_result = rsa_array(data, [model_dsm, model_dsm],
                               temporal_radius=1, spatial_radius=None,
                               data_dsm_metric='euclidean')
        assert rsa_result.shape == (1, 2, 2)
        assert_equal(rsa_result, 1)

        # One model DSM, paralellization across 2 CPUs
        rsa_result = rsa_array(data, model_dsm,
                               temporal_radius=1, spatial_radius=None,
                               data_dsm_metric='euclidean', n_jobs=2)
        assert rsa_result.shape == (1, 2)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, paralellization across 2 CPUs
        rsa_result = rsa_array(data, [model_dsm, model_dsm],
                               temporal_radius=1, spatial_radius=None,
                               data_dsm_metric='euclidean', n_jobs=2)
        assert rsa_result.shape == (1, 2, 2)
        assert_equal(rsa_result, 1)

    def test_rsa_spat(self):
        """Test RSA with a spatial searchlight."""
        data = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
        model_dsm = np.array([1, 2, 1])
        dist = np.array([[0, 1, 2],
                         [1, 0, 1],
                         [2, 1, 0]])

        # One model DSM, no paralellization
        rsa_result = rsa_array(data, model_dsm, dist,
                               temporal_radius=None, spatial_radius=1,
                               data_dsm_metric='euclidean')
        assert rsa_result.shape == (3, 1)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, no paralellization
        rsa_result = rsa_array(data, [model_dsm, model_dsm], dist,
                               temporal_radius=None, spatial_radius=1,
                               data_dsm_metric='euclidean')
        assert rsa_result.shape == (3, 1, 2)
        assert_equal(rsa_result, 1)

        # One model DSM, paralellization across 2 CPUs
        rsa_result = rsa_array(data, model_dsm, dist,
                               temporal_radius=None, spatial_radius=1,
                               data_dsm_metric='euclidean', n_jobs=2)
        assert rsa_result.shape == (3, 1)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, paralellization across 2 CPUs
        rsa_result = rsa_array(data, [model_dsm, model_dsm], dist,
                               temporal_radius=None, spatial_radius=1,
                               data_dsm_metric='euclidean', n_jobs=2)
        assert rsa_result.shape == (3, 1, 2)
        assert_equal(rsa_result, 1)

    def test_rsa_spat_temp(self):
        """Test RSA with a spatial-temporal searchlight."""
        data = np.array([[[1, 2, 3], [2, 3, 4]],
                         [[2, 3, 4], [3, 4, 5]],
                         [[3, 4, 5], [4, 5, 6]]])
        model_dsm = np.array([1, 2, 1])
        dist = np.array([[0, 1, 2],
                         [1, 0, 1],
                         [2, 1, 0]])

        # One model DSM, no paralellization
        rsa_result = rsa_array(data, model_dsm, dist,
                               temporal_radius=1, spatial_radius=1,
                               data_dsm_metric='euclidean')
        assert rsa_result.shape == (2, 1)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, no paralellization
        rsa_result = rsa_array(data, [model_dsm, model_dsm], dist,
                               temporal_radius=1, spatial_radius=1,
                               data_dsm_metric='euclidean')
        assert rsa_result.shape == (2, 1, 2)
        assert_equal(rsa_result, 1)

        # One model DSM, paralellization across 2 CPUs
        rsa_result = rsa_array(data, model_dsm, dist,
                               temporal_radius=1, spatial_radius=1,
                               data_dsm_metric='euclidean', n_jobs=2)
        assert rsa_result.shape == (2, 1)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, paralellization across 2 CPUs
        rsa_result = rsa_array(data, [model_dsm, model_dsm], dist,
                               temporal_radius=1, spatial_radius=1,
                               data_dsm_metric='euclidean', n_jobs=2)
        assert rsa_result.shape == (2, 1, 2)
        assert_equal(rsa_result, 1)
