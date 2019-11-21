import pytest
from types import GeneratorType
import numpy as np
from rsa import rsa_gen


def dsm():
    """Create an example DSM"""
    return np.array([1, 2, 3, 4, 5, 6])


def dsm_gen(dsms):
    """Generator for DSMs"""
    for dsm in dsms:
        yield dsm


class TestRsaGen:
    """Test the rsa_gen function"""

    def test_return_type(self):
        """Test return type of rsa_gen"""
        assert isinstance(rsa_gen(dsm_gen([dsm()]), dsm()), GeneratorType)
        assert next(rsa_gen(dsm_gen([dsm()]), dsm())).shape == tuple()
        assert next(rsa_gen(dsm_gen([dsm()]), [dsm()])).shape == (1,)

    def test_spearman(self):
        """Test computing RSA with Spearman correlation"""
        data_dsm = dsm_gen([np.array([1, 2, 3])])
        model_dsm = np.array([2, 3, 3.5])
        assert next(rsa_gen(data_dsm, model_dsm, metric='spearman')) == 1.0

    def test_pearson(self):
        """Test computing RSA with Pearson correlation"""
        data_dsm = dsm_gen([np.array([1, 2, 3])])
        model_dsm = np.array([2, 3, 3.5])
        assert next(rsa_gen(data_dsm, model_dsm, metric='pearson')) == -0.5
