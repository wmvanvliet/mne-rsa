__version__ = '0.4'
from .source_level import rsa_stcs, dsm_stcs, rsa_nifti, dsm_nifti
from .sensor_level import rsa_evokeds, rsa_epochs, dsm_evokeds, dsm_epochs
from .searchlight import searchlight
from .rsa import rsa, rsa_gen, rsa_array
from .dsm import compute_dsm, compute_dsm_cv, dsm_array
from .viz import plot_dsms, plot_dsms_topo
