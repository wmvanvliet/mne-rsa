Representational Similarity Analysis
------------------------------------

This is a Python package for performing representational similarity analysis (RSA) using [MNE-Python](https://martinos.org/mne/stable/index.html) data structures. The RSA is computed using a "searchlight" approach.

## Installation

Here is how to install the package as a user:

`pip install git+https://github.com/wmvanvliet/mne-rsa.git`


## Use cases

This is what the package can do for you:

 - Compute DSMs on arbitrary data
 - Compute DSMs in a searchlight across:
    - vertices and samples (source level)
    - sensors and samples (sensor level)
    - vertices only (source level)
    - sensors only (sensor level)
    - samples only (source and sensor level)
 - Use cross-validated distance metrics when computing DSMs
 - And of course: compute RSA between DSMs

This is what it cannot do (yet) for you:

 - Compute DSMs in a searchlight across voxels (volume level)

Supported metrics for comparing DSMs:

  - Spearman correlation (the default)
  - Pearson correlation
  - Kendall's Tau-A
  - Linear regression (when comparing multiple DSMs at once)
  - Partial correlation (when comparing multiple DSMs at once)


## Juicy bits of the API 

```python
def compute_dsm(model, pca=False, metric='correlation', **kwargs)

def rsa_source_level(stcs, model_dsm, src, y=None,
                     spatial_radius=0.04, temporal_radius=0.1,
		     stc_dsm_metric='correlation', stc_dsm_params=None,
                     rsa_metric='spearman',
                     n_jobs=1, verbose=False)

def rsa_evokeds(evokeds, model_dsm, y=None, noise_cov=None,
                spatial_radius=0.04, temporal_radius=0.1,
                evoked_dsm_metric='correlation', evoked_dsm_params=None,
                rsa_metric='spearman',
                n_jobs=1, verbose=False)

def rsa_epochs(epochs, model_dsm, y=None, noise_cov=None,
               spatial_radius=0.04, temporal_radius=0.1,
               epochs_dsm_metric='correlation', epochs_dsm_params=None,
               rsa_metric='spearman',
               n_jobs=1, verbose=False)
```

## Example usage

Basic example on the EEG "kiloword" data:

```python
import mne
import rsa
data_path = mne.datasets.kiloword.data_path(verbose=True)
epochs = mne.read_epochs(data_path + '/kword_metadata-epo.fif')
# Compute the model DSM using all word properties
dsm_model = rsa.compute_dsm(epochs.metadata.iloc[:, 1:].values)
evoked_rsa = rsa.rsa_epochs(epochs, dsm_model,
                            spatial_radius=0.04, temporal_radius=0.01,
                            verbose=True)
```


## Integration with other packages

I mainly wrote this package to perform RSA analysis in MEG data. Hence, integration functions with [MNE-Python](https://mne.tools) are provided. No integration with [nipy](https://nipy.org) yet for fMRI, feel free to submit a PR!


## Performance

This package aims to be fast and memory efficient. An important design feature is that under the hood, everything operates on generators. The searchlight routines produce a generator of DSMs which are consumed by a generator of RSA values. Parallel processing is also supported, so you can use all of your CPU cores.


## Development

Here is how to set up the package as a developer:

```
git clone git@github.com:wmvanvliet/mne-rsa.git
cd mne-rsa
python setup.py develop --user
```
