import pytest
from types import GeneratorType
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from mne_rsa import searchlight, rsa, rsa_gen, rsa_array
from mne_rsa.rsa import _kendall_tau_a, _partial_correlation


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
        assert next(rsa_gen(data_dsm, model_dsm, metric="spearman")) == 1.0

        data_dsm = dsm_gen([[np.NaN, 2, 3, 4, 5]])
        model_dsm = np.array([2, np.NaN, 3.5, 4, 5])
        assert (
            next(rsa_gen(data_dsm, model_dsm, metric="spearman", ignore_nan=True))
            == 1.0
        )

    def test_pearson(self):
        """Test computing RSA with Pearson correlation"""
        data_dsm = dsm_gen([[1, 2, 3]])
        model_dsm = np.array([2, 3, 3.5])
        assert next(rsa_gen(data_dsm, model_dsm, metric="pearson")) < 1.0

        data_dsm = dsm_gen([[np.NaN, 2, 3, 4, 5]])
        model_dsm = np.array([2, np.NaN, 3.5, 4, 5])
        assert (
            next(rsa_gen(data_dsm, model_dsm, metric="pearson", ignore_nan=True)) < 1.0
        )

    def test_kendall_tau_a(self):
        """Test computing RSA with Kendall's Tau Alpha"""
        data_dsm = dsm_gen([[1, 2, 3]])
        model_dsm = np.array([1, 3, 3])  # This metric deals well with ties
        rsa_val = next(rsa_gen(data_dsm, model_dsm, metric="kendall-tau-a"))
        assert rsa_val == 2 / 3

        data_dsm = dsm_gen([[1, np.NaN, 2, 3, 4]])
        model_dsm = np.array([1, 2, 3, 3, np.NaN])
        rsa_val = next(
            rsa_gen(data_dsm, model_dsm, metric="kendall-tau-a", ignore_nan=True)
        )
        assert rsa_val == 2 / 3

    def test_regression(self):
        """Test computing RSA with regression."""
        model_dsm1 = np.array([-1, 0, 1])
        model_dsm2 = np.array([1, -2, 1])
        data_dsm = dsm_gen([3 * model_dsm1 + 5 * model_dsm2])
        rsa_val = next(rsa_gen(data_dsm, [model_dsm1, model_dsm2], metric="regression"))
        assert_allclose(rsa_val, [3, 5])

        model_dsm1 = np.array([-1, np.NaN, 0, 1, 1])
        model_dsm2 = np.array([1, 3, -2, 1, np.NaN])
        data_dsm = dsm_gen([3 * model_dsm1 + 5 * model_dsm2])
        rsa_val = next(
            rsa_gen(
                data_dsm, [model_dsm1, model_dsm2], metric="regression", ignore_nan=True
            )
        )
        assert_allclose(rsa_val, [3, 5])

    def test_partial(self):
        """Test computing RSA with partial correlation."""
        # Example taken from https://en.wikipedia.org/wiki/Partial_correlation
        model_dsm1 = np.array([1, 2, 3, 4])
        model_dsm2 = np.array([0, 0, 1, 1])
        data_dsm = dsm_gen([[2, 4, 15, 20]])
        rsa_val = next(rsa_gen(data_dsm, [model_dsm1, model_dsm2], metric="partial"))
        assert_allclose(rsa_val, [0.919145, 0.912871])

        model_dsm1 = np.array([1, np.NaN, 2, 3, 4, 4])
        model_dsm2 = np.array([0, 0, 0, 1, 1, np.NaN])
        data_dsm = dsm_gen([[2, np.NaN, 4, 15, 20, np.NaN]])
        rsa_val = next(
            rsa_gen(
                data_dsm, [model_dsm1, model_dsm2], metric="partial", ignore_nan=True
            )
        )
        assert_allclose(rsa_val, [0.919145, 0.912871])

    def test_partial_spearman(self):
        """Test computing RSA with partial spearman correlation."""
        # Example verified with MATLAB's partialcorr function
        model_dsm1 = np.array([1, 2, 3, 4])
        model_dsm2 = np.array([0, 0, 1, 1])
        data_dsm = dsm_gen([[2, 4, 20, 15]])
        rsa_val = next(
            rsa_gen(data_dsm, [model_dsm1, model_dsm2], metric="partial-spearman")
        )
        assert_allclose(rsa_val, [0, 2 / 3], atol=1e-15)

        model_dsm1 = np.array([1, np.NaN, 2, 3, 4, 4])
        model_dsm2 = np.array([0, 0, 0, 1, 1, np.NaN])
        data_dsm = dsm_gen([[2, np.NaN, 4, 20, 15, np.NaN]])
        rsa_val = next(
            rsa_gen(
                data_dsm,
                [model_dsm1, model_dsm2],
                metric="partial-spearman",
                ignore_nan=True,
            )
        )
        assert_allclose(rsa_val, [0, 2 / 3], atol=1e-15)

    def test_invalid_metric(self):
        """Test whether computing RSA with an invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid RSA metric"):
            next(rsa_gen(dsm_gen([dsm()]), dsm(), metric="foo"))

        # These metrics only work with multiple model DSMs
        with pytest.raises(ValueError, match="Need more than one model DSM"):
            next(rsa_gen(dsm_gen([dsm()]), dsm(), metric="partial"))
        with pytest.raises(ValueError, match="Need more than one model DSM"):
            next(rsa_gen(dsm_gen([dsm()]), dsm(), metric="partial-spearman"))

    def test_nan(self):
        """Test whether NaNs generate an error when appropriate."""
        assert np.isnan(next(rsa_gen(dsm_gen([[1, 2, np.NaN, 4, 5, 6]]), dsm())))
        assert_allclose(
            next(rsa_gen(dsm_gen([[1, 2, np.NaN, 4, 5, 6]]), dsm(), ignore_nan=True)),
            1,
            atol=1e-15,
        )


class TestRSA:
    """Test the main RSA function"""

    # Most of the functionality is already tested in TestRSAGen
    def test_return_type(self):
        """Test return type of rsa_gen"""
        assert isinstance(rsa([dsm()], dsm()), np.ndarray)
        assert rsa(dsm(), dsm()).shape == tuple()
        assert rsa(dsm(), [dsm()]).shape == (1,)

    def test_progress_bar(self):
        """Test showing a progress bar for rsa"""
        assert rsa([dsm()], dsm(), verbose=True) == [1.0]
        assert rsa([dsm()], dsm(), verbose=True, n_data_dsms=1) == [1.0]
        assert rsa(dsm_gen([dsm()]), dsm(), verbose=True) == [1.0]

    def test_parallel_processing(self):
        """Test running rsa with multiple cores."""
        assert_equal(rsa([dsm(), dsm()], dsm(), n_jobs=2), [1.0, 1.0])


class TestRSASearchlight:
    """Test computing RSA using searchlight patches."""

    def test_invalid_input(self):
        """Test invalid inputs."""
        data = np.array([[1], [2], [3], [4]])
        model_dsm = dsm()
        with pytest.raises(ValueError, match="There is only a single feature"):
            rsa_array(data, model_dsm)

    def test_rsa_single_searchlight_patch(self):
        """Test RSA with a single searchlight patch."""
        data = np.array([[1], [2], [3], [4]])
        model_dsm = np.array([1, 2, 3, 1, 2, 1])
        rsa_result = rsa_array(data, model_dsm, data_dsm_metric="euclidean")
        assert rsa_result.shape == tuple()  # Single scalar value
        assert rsa_result == 1

    def test_rsa_temp(self):
        """Test RSA with a temporal searchlight."""
        data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
        model_dsm = np.array([1, 2, 1])

        # One model DSM, no parallelization
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(data, model_dsm, patches, data_dsm_metric="euclidean")
        assert rsa_result.shape == (2,)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, no parallelization
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(
            data, [model_dsm, model_dsm], patches, data_dsm_metric="euclidean"
        )
        assert rsa_result.shape == (2, 2)
        assert_equal(rsa_result, 1)

        # One model DSM, parallelization across 2 CPUs
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(
            data, model_dsm, patches, data_dsm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2,)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, parallelization across 2 CPUs
        patches = searchlight(data.shape, temporal_radius=1)
        rsa_result = rsa_array(
            data, [model_dsm, model_dsm], patches, data_dsm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2, 2)
        assert_equal(rsa_result, 1)

    def test_rsa_spat(self):
        """Test RSA with a spatial searchlight."""
        data = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
        model_dsm = np.array([1, 2, 1])
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        # One model DSM, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(data, model_dsm, patches, data_dsm_metric="euclidean")
        assert rsa_result.shape == (3,)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(
            data, [model_dsm, model_dsm], patches, data_dsm_metric="euclidean"
        )
        assert rsa_result.shape == (3, 2)
        assert_equal(rsa_result, 1)

        # One model DSM, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(
            data, model_dsm, patches, data_dsm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (3,)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1)
        rsa_result = rsa_array(
            data, [model_dsm, model_dsm], patches, data_dsm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (3, 2)
        assert_equal(rsa_result, 1)

    def test_rsa_spat_temp(self):
        """Test RSA with a spatial-temporal searchlight."""
        data = np.array(
            [[[1, 2, 3], [2, 3, 4]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 6]]]
        )
        model_dsm = np.array([1, 2, 1])
        dist = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])

        # One model DSM, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(data, model_dsm, patches, data_dsm_metric="euclidean")
        assert rsa_result.shape == (2, 1)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, no paralellization
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(
            data, [model_dsm, model_dsm], patches, data_dsm_metric="euclidean"
        )
        assert rsa_result.shape == (2, 1, 2)
        assert_equal(rsa_result, 1)

        # One model DSM, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(
            data, model_dsm, patches, data_dsm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2, 1)
        assert_equal(rsa_result, 1)

        # Multiple model DSMs, paralellization across 2 CPUs
        patches = searchlight(data.shape, dist, spatial_radius=1, temporal_radius=1)
        rsa_result = rsa_array(
            data, [model_dsm, model_dsm], patches, data_dsm_metric="euclidean", n_jobs=2
        )
        assert rsa_result.shape == (2, 1, 2)
        assert_equal(rsa_result, 1)


class TestKendallTau:
    def test_basic(self):
        """Test computing Kendall's Tau Alpha"""
        # This metric deals well with ties
        assert _kendall_tau_a([1, 2, 3], [1, 3, 3]) == 2 / 3

        # Test taken from scipy
        assert _kendall_tau_a([9, 2, 5, 6], [4, 7, 9, 11]) == 0

    def test_sizes(self):
        """Test feeding arrays to Kendall's Tau Alpha of unequal size."""
        with pytest.raises(ValueError, match="must be of the same size"):
            _kendall_tau_a([1, 2, 3], [1, 2, 3, 4, 5])

    def test_empty(self):
        """Test feeding empty arrays to Kendall's Tau Alpha."""
        assert np.isnan(_kendall_tau_a([], []))

    def test_only_ties(self):
        """Test Kendall's Tau Alpha where every value is a tie."""
        assert np.isnan(_kendall_tau_a([1, 1, 1], [2, 2, 2]))


class TestPartialCorrelation:
    def test_basic(self):
        """Test computing partial correlations."""
        data_dsm = [2, 4, 15, 20]
        model_dsms = [[1, 2, 3, 4], [0, 0, 1, 1]]
        assert_allclose(
            _partial_correlation(data_dsm, model_dsms, type="pearson"),
            [0.919145, 0.912871],
        )
        print(_partial_correlation(data_dsm, model_dsms, type="spearman"))
        assert_allclose(
            _partial_correlation(data_dsm, model_dsms, type="spearman"),
            [-1, 0.89442719],
        )

    def test_sizes(self):
        """Test feeding arrays of invalid size to _partial_correlation."""
        with pytest.raises(ValueError, match="Need more than one model DSM"):
            _partial_correlation([1, 2], [[1, 2]])
        with pytest.raises(ValueError, match="Correlation type must be"):
            _partial_correlation([1, 2], [[1, 2], [1, 2]], type="banana")
