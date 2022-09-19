#!/usr/bin/env python
# coding: utf-8

"""
Using cross-validation when computing DSMs
==========================================

This example demonstrates how to perform cross-validation when computing
dissimilarity matrices (DSMs). When the data has repeated measurements of the
same stimulus type, cross-validation can be used to provide much more robust
distance estimates between stimulus types. Repeated measurements can for
example be actual repetitions of the same stimulus within the same recording,
or recordings on multiple volunteers with the same stimuli.

The dataset will be the kiloword dataset [1]_: approximately 1,000 words were
presented to 75 participants in a go/no-go lexical decision task while
event-related potentials (ERPs) were recorded.

This dataset as provided does not have repeated measurements of the same
stimuli. To illustrate cross-validation, we will treat words with the same
number of letters as being repeated measurements of the same stimulus type.

.. [1] Dufau, S., Grainger, J., Midgley, KJ., Holcomb, PJ (2015). A thousand
       words are worth a picture: Snapshots of printed-word processing in an
       event-related potential megastudy. Psychological science.
"""

# Import required packages
import mne
import mne_rsa

###############################################################################
# MNE-Python contains a build-in data loader for the kiloword dataset. We use
# it here to read it as 960 epochs. Each epoch represents the brain response to
# a single word, averaged across all the participants. For this example, we
# speed up the computation, at a cost of temporal precision, by downsampling
# the data from the original 250 Hz. to 100 Hz.

data_path = mne.datasets.kiloword.data_path(verbose=True)
epochs = mne.read_epochs(data_path / 'kword_metadata-epo.fif')
epochs = epochs.resample(100)


###############################################################################
# The ``epochs`` object contains a ``.metadata`` field that contains
# information about the 960 words that were used in the experiment. Let's have
# a look at the metadata for the 10 random words:

epochs.metadata.sample(10)


###############################################################################
# The kiloword dataset as provided does not have repeated measurements of the
# same stimuli. To illustrate cross-validation, we will treat words with the
# same number of letters as being repeated measurements of the same stimulus
# type.
#
# To denote which epochs are repetitions of the same stimulus, we create a list
# ``y`` that contains integer labels for each epoch. Repetitions of the same
# stimulus have the same label in the ``y`` list. This scheme is lifted from
# the machine learning literature (and the Scikit-Learn API). In this example,
# we use the number of letters in the stimulus words as the labels for the
# epochs.

y = epochs.metadata.NumberOfLetters.astype(int)


###############################################################################
# Many high-level functions in the MNE-RSA module can take the ``y`` list as a
# parameter to enable cross-validation. Notably the functions for performing
# RSA and computing DSMs. In this example, we will restrict the analysis to
# computing DSMs using a spatio-temporal searchlight on the sensor-level data.

dsms = mne_rsa.dsm_epochs(
    epochs,                      # The EEG data
    y=y,                         # Set labels to enable cross validation         
    n_folds=5,                   # Number of folds to use during cross validation
    dist_metric='sqeuclidean',   # Distance metric to compute the DSMs
    spatial_radius=0.45,         # Spatial radius of the searchlight patch in meters.
    temporal_radius=0.05,        # Temporal radius of the searchlight path in seconds.
    tmin=0.15, tmax=0.25)        # To save time, only analyze this time interval


###############################################################################
# Plotting the cross-validated DSMs

mne_rsa.plot_dsms_topo(dsms, epochs.info)


###############################################################################
# For performance reasons, the low-level functions of MNE-RSA do not take a
# ``y`` list for cross-validation. Instead, they require the data to be already
# split into folds. The :func:`create_folds` function can create these
# folds.

X = epochs.get_data()
y = epochs.metadata.NumberOfLetters.astype(int)
folds = mne_rsa.create_folds(X, y, n_folds=5)

dsm = mne_rsa.compute_dsm_cv(folds, metric='euclidean')
mne_rsa.plot_dsms(dsm)
