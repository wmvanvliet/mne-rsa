#!/usr/bin/env python
# coding: utf-8

"""
Source-level RSA using a searchlight
====================================

This example demonstrates how to perform representational similarity analysis
(RSA) on source localized MEG data, using a searchlight approach.

In the searchlight approach, representational similarity is computed between
the model and searchlight "patches". A patch is defined by a seed vertex on the
cortex and all vertices within a given radius. By default, patches are created
using each vertex as a seed point, so you can think of it as a "searchlight"
that scans along the cortex.

The radius of a searchlight can be defined in space, in time, or both. In
this example, our searchlight will have a spatial radius of 2 cm. and a
temporal radius of 20 ms.

The dataset will be the MNE-sample dataset: a collection of 288 epochs in which
the participant was presented with an auditory beep or visual stimulus to
either the left or right ear or visual field.
"""
# sphinx_gallery_thumbnail_number=2

# Import required packages
import os.path as op
import mne
import mne_rsa

sample_path = mne.datasets.sample.data_path()
testing_path = mne.datasets.testing.data_path()
data_path = op.join(sample_path, 'MEG', 'sample')
subjects_dir = op.join(sample_path, 'subjects')

mne.set_log_level(False)  # Be less verbose

raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_filt-0-40_raw.fif'))
events = mne.read_events(op.join(data_path, 'sample_audvis_filt-0-40_raw-eve.fif'))
event_id = {'audio/left': 1,
            'audio/right': 2,
            'visual/left': 3,
            'visual/right': 4}
epochs = mne.Epochs(raw, events, event_id, preload=True)
epochs.resample(100)
epochs = mne.concatenate_epochs([epochs[cl] for cl in ['audio/left', 'audio/right', 'visual/left', 'visual/right']])


def sensitivity_metric(event_id_1, event_id_2):
    if event_id_1 == 1 and event_id_2 == 1:
        return 0
    if event_id_1 == 2 and event_id_2 == 2:
        return 0.5
    elif event_id_1 == 1 and event_id_2 == 2:
        return 0.5
    elif event_id_1 == 2 and event_id_1 == 1:
        return 0.5
    else:
        return 1


model_dsm = mne_rsa.compute_dsm(epochs.events[:, 2], metric=sensitivity_metric)
mne_rsa.plot_dsms(model_dsm, title='Model DSM')

inv = mne.minimum_norm.read_inverse_operator(
    op.join(testing_path, 'MEG', 'sample', 'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif'))
epochs_stc = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=0.1111)

rsa_vals = mne_rsa.rsa_source_level(
    epochs_stc,
    model_dsm,
    inv['src'],
    stc_dsm_metric='correlation',
    spatial_radius=0.02,
    temporal_radius=0.02,
    tmin=0, tmax=0.2,
    n_jobs=1, verbose=True,
)

_, peak_time = rsa_vals.get_peak()
rsa_vals.plot('sample', subjects_dir=subjects_dir, initial_time=peak_time)
