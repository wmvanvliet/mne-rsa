__version__ = '0.4.dev0'
from .source_level import rsa_source_level, dsm_source_level
from .sensor_level import rsa_evokeds, rsa_epochs, dsm_evokeds, dsm_epochs
from .searchlight import searchlight
from .rsa import rsa, rsa_gen, rsa_array
from .dsm import compute_dsm, compute_dsm_cv, dsm_array
from .viz import plot_dsms, plot_dsms_topos