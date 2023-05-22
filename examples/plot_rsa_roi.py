#!/usr/bin/env python
# coding: utf-8
"""
Source-level RSA using ROI's
============================

In this example, we use anatomical labels as Regions Of Interest (ROIs). Rather
than using a searchlight, we compute DSMs for each ROI and then compute RSA
with a single model DSM.

The dataset will be the MNE-sample dataset: a collection of 288 epochs in which
the participant was presented with an auditory beep or visual stimulus to
either the left or right ear or visual field.
"""
# sphinx_gallery_thumbnail_number=2
# Import required packages
import mne
import mne_rsa

mne.set_log_level(True)  # Be less verbose
mne.viz.set_3d_backend("pyvista")

###############################################################################
# We'll be using the data from the MNE-sample set. To speed up computations in
# this example, we're going to use one of the sparse source spaces from the
# testing set.
sample_root = mne.datasets.sample.data_path(verbose=True)
testing_root = mne.datasets.testing.data_path(verbose=True)
sample_path = sample_root / "MEG" / "sample"
testing_path = testing_root / "MEG" / "sample"
subjects_dir = sample_root / "subjects"

###############################################################################
# Creating epochs from the continuous (raw) data. We downsample to 100 Hz to
# speed up the RSA computations later on.
raw = mne.io.read_raw_fif(sample_path / "sample_audvis_filt-0-40_raw.fif")
events = mne.read_events(sample_path / "sample_audvis_filt-0-40_raw-eve.fif")
event_id = {"audio/left": 1, "audio/right": 2, "visual/left": 3, "visual/right": 4}
epochs = mne.Epochs(raw, events, event_id, preload=True)
epochs.resample(100)

###############################################################################
# It's important that the model DSM and the epochs are in the same order, so
# that each row in the model DSM will correspond to an epoch. The model DSM
# will be easier to interpret visually if the data is ordered such that all
# epochs belonging to the same experimental condition are right next to
# each-other, so patterns jump out. This can be achieved by first splitting the
# epochs by experimental condition and then concatenating them together again.
epoch_splits = [
    epochs[cl] for cl in ["audio/left", "audio/right", "visual/left", "visual/right"]
]
epochs = mne.concatenate_epochs(epoch_splits)

###############################################################################
# Now that the epochs are in the proper order, we can create a DSM based on the
# experimental conditions. This type of DSM is referred to as a "sensitivity
# DSM". Let's create a sensitivity DSM that will pick up the left auditory
# response when RSA-ed against the MEG data. Since we want to capture areas
# where left beeps generate a large signal, we specify that left beeps should
# be similar to other left beeps. Since we do not want areas where visual
# stimuli generate a large signal, we specify that beeps must be different from
# visual stimuli. Furthermore, since in areas where visual stimuli generate
# only a small signal, random noise will dominate, we also specify that visual
# stimuli are different from other visual stimuli. Finally left and right
# auditory beeps will be somewhat similar.


def sensitivity_metric(event_id_1, event_id_2):
    """Determine similarity between two epochs, given their event ids."""
    if event_id_1 == 1 and event_id_2 == 1:
        return 0  # Completely similar
    if event_id_1 == 2 and event_id_2 == 2:
        return 0.5  # Somewhat similar
    elif event_id_1 == 1 and event_id_2 == 2:
        return 0.5  # Somewhat similar
    elif event_id_1 == 2 and event_id_1 == 1:
        return 0.5  # Somewhat similar
    else:
        return 1  # Not similar at all


model_dsm = mne_rsa.compute_dsm(epochs.events[:, 2], metric=sensitivity_metric)
mne_rsa.plot_dsms(model_dsm, title="Model DSM")

###############################################################################
# This example is going to be on source-level, so let's load the inverse
# operator and apply it to obtain a cortical surface source estimate for each
# epoch. To speed up the computation, we going to load an inverse operator from
# the testing dataset that was created using a sparse source space with not too
# many vertices.
inv = mne.minimum_norm.read_inverse_operator(
    f"{testing_path}/sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif"
)
epochs_stc = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=0.1111)

###############################################################################
# ROIs need to be defined as ``mne.Label`` objects. Here, we load the APARC
# parcellation generated by FreeSurfer and treat each parcel as an ROI.
rois = mne.read_labels_from_annot(
    parc="aparc", subject="sample", subjects_dir=subjects_dir
)

###############################################################################
# Performing the RSA. To save time, we don't use a searchlight over time, just
# over the ROIs. The results are returned not only as a NumPy `ndarray`, but
# also as an `mne.SourceEstimate` object, where each vertex beloning to the
# same ROI has the same value.
rsa_vals, stc = mne_rsa.rsa_stcs_rois(
    epochs_stc,
    model_dsm,
    inv["src"],
    rois,
    temporal_radius=None,
    n_jobs=1,
    verbose=False,
)

###############################################################################
# To plot the RSA values on a brain, we can use one of MNE-RSA's own
# visualization functions.
brain = mne_rsa.plot_roi_map(
    rsa_vals, rois, subject="sample", subjects_dir=subjects_dir
)
brain.show_view("lateral", distance=600)
