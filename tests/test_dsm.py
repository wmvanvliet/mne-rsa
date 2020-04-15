import numpy as np

from mne_rsa import dsm_temp

class TestDsmTemp:
    """Test computing DSMs with temporal searchlight patches."""
    def test_basic(self):
        data = np.array([[1, 2, 3], [1, 2, 3]])
        dsms = dsm_temp(data, temporal_radius=1, dist_metric='euclidean')
        assert list(dsms) == [0]
