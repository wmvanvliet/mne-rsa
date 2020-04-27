#!/usr/bin/env python
# coding: utf-8

"""
Construct a model DSM
=====================

This example shows how to create DSMs from arbitrary data. A common use case
for this is to construct a "model" DSM to RSA against the brain data. In this
example, we will create a DSM based on the length of the words shown during an
EEG experiment.
"""

# Import required packages
import mne
import mne_rsa

###############################################################################
# MNE-Python contains a build-in data loader for the kiloword dataset, which is
# used here as an example dataset. Since we only need the words shown during
# the experiment, which are in the metadata, we can pass ``preload=False`` to
# prevent MNE-Python from loading the EEG data, which is a nice speed gain.

data_path = mne.datasets.kiloword.data_path(verbose=True)
epochs = mne.read_epochs(data_path + '/kword_metadata-epo.fif', preload=False)

# Show the metadata of 10 random epochs
epochs.metadata.sample(10)

###############################################################################
# Now we are ready to create the "model" DSM, which will encode the difference
# in length between the words shown during the experiment.

dsm = mne_rsa.compute_dsm(epochs.metadata.NumberOfLetters, metric='euclidean')

# Plot the DSM
fig = mne_rsa.plot_dsms(dsm, title='Word length DSM')
fig.set_size_inches(3, 3)  # Make figure a little bigger to show axis properly
