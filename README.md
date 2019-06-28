Representational Similarity Analysis
------------------------------------

This is a Python package for performining representational similarity analysis (RSA) using [MNE-Python](https://martinos.org/mne/stable/index.html) data structures.

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

We actually encountered these:
 - Compute RSA across vertices and samples (source level)
 - Compute RSA across sensors and samples (sensor level)
 - Compute RSA across voxels (volume level)
 - Compute statistics

Possible other scenarios we haven't encountered yet:
 - Compute RSA across vertices only (source level)
 - Compute RSA across sensors only (sensor level)
 - Compute RSA across samples only (source and sensor level)


## Parameters

 - Brain data:
   - MNE data structures: list of Evokeds, (list of SourceEstimate + 1 SourceSpaces)
   - 2D or 3D NumPy array (n_items x n_space_things x n_time_things) + positions (time is always consecutive)
 - Norm data: 1D or 2D NumPy array (n_items x n_features)
 - Distance metric for creating the DSMs
 - Distance metric for comparing two DSMs
 - Radius of the searchlight (time, space, space-time)
 - Number of processes to run in parallel
 - Verbosity (display progressbar or not?)

Return values:
 - MNE data structures: Evoked (sensor level), SourceEstimate (source level)
 - Numpy array (voxel data) same shape as the input data


## Suggested API 

```python
def rsa_stcs(stcs, model, src, stc_dsm_metric='correlation',
             model_dsm_metric='correlation', rsa_metric='spearman',
             spatial_radius=0, temporal_radius=0, break_after=-1,
             n_jobs=1, verbose=False)

def rsa_evokeds(evokeds, model, evoked_dsm_metric='correlation',
                model_dsm_metric='correlation', rsa_metric='spearman',
                spatial_radius=0, temporal_radius=0, break_after=-1,
                n_jobs=1, verbose=False)

def rsa_array(data, model, positions, data_dsm_metric='correlation',
              model_dsm_metric='correlation', rsa_metric='spearman',
              spatial_radius=0, temporal_radius=0, break_after=-1,
              n_jobs=1, verbose=False)
```


## To think about

How to deal with gradiometer pairs? (combine them first?)
How to deal with mixed sensor types? (just give error when present in data, normalize the data)


## Documentation

 - Docstrings
 - Comments in code
 - Examples


## Tests

 - Unit tests
