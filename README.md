Representational Similarity Analysis
------------------------------------

This is a Python package for performing representational similarity analysis (RSA) using [MNE-Python](https://martinos.org/mne/stable/index.html) data structures. The RSA is computed using a "searchlight" approach.

## Installation

Here is how to install the package as a user:

`pip install git+https://github.com/wmvanvliet/rsa.git`


## Development

Here is how to set up the package as a developer:

```
git clone git@github.com:wmvanvliet/rsa.git
cd rsa
python setup.py develop --user
```


## Use cases

This is what the package can do for you:

 - Compute RSA across vertices and samples (source level)
 - Compute RSA across sensors and samples (sensor level)
 - Compute RSA across vertices only (source level)
 - Compute RSA across sensors only (sensor level)
 - Compute RSA across samples only (source and sensor level)

This is what it cannot do (yet) for you:

 - Compute RSA across voxels (volume level)


## Juicy bits of the API 

```python
def rsa_source_level(stcs, model, src,
                     spatial_radius=0.04, temporal_radius=0.1,
		     stc_dsm_metric='correlation', model_dsm_metric='correlation', rsa_metric='spearman',
                     n_jobs=1, verbose=False)

def rsa_evokeds(evokeds, model, noise_cov=None,
		spatial_radius=0.04, temporal_radius=0.1,
                evoked_dsm_metric='correlation', model_dsm_metric='correlation', rsa_metric='spearman',
                n_jobs=1, verbose=False)
```

## Example usage

Basic example on the EEG "kiloword" data:

```python
import mne
import rsa
data_path = mne.datasets.kiloword.data_path(verbose=True)
epochs = mne.read_epochs(data_path + '/kword_metadata-epo.fif')
evokeds = [epochs[w].average() for w in epochs.metadata['WORD']]
model = epochs.metadata.iloc[:, 1:].values  # All word properties
evoked_rsa = rsa.rsa_evokeds(evokeds, model,
                             spatial_radius=0.04, temporal_radius=0.01,
			     verbose=True)
```
