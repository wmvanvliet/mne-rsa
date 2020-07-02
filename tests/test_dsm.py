import numpy as np
from numpy.testing import assert_equal

from mne_rsa import searchlight, dsms_array


class TestDsmsSearchlight:
    """Test computing DSMs with searchlight patches."""

    def test_temporal(self):
        data = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        patches = searchlight(data.shape, temporal_radius=1)
        dsms = dsms_array(data, patches, dist_metric='euclidean')
        assert len(dsms) == 2
        assert dsms.shape == (2, 1)
        assert_equal(list(dsms), [0, 0])
