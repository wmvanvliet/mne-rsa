#!/usr/bin/env python
# coding: utf-8

"""
Compute RSA between DSMs
========================

This example showcases the most basic version of RSA: computing the similarity
between two DSMs. Then we continue with computing RSA between many DSMs
efficiently.
"""

# Import required packages
import pandas as pd
from matplotlib import pyplot as plt
import mne
import mne_rsa

###############################################################################
# MNE-Python contains a build-in data loader for the kiloword dataset, which is
# used here as an example dataset. Since we only need the words shown during
# the experiment, which are in the metadata, we can pass ``preload=False`` to
# prevent MNE-Python from loading the EEG data, which is a nice speed gain.

data_path = mne.datasets.kiloword.data_path(verbose=True)
epochs = mne.read_epochs(data_path + '/kword_metadata-epo.fif')

# Show the metadata of 10 random epochs
epochs.metadata.sample(10)

###############################################################################
# Compute DSMs based on word length and visual complexity.

metadata = epochs.metadata
dsm1 = mne_rsa.compute_dsm(metadata.NumberOfLetters, metric='euclidean')
dsm2 = mne_rsa.compute_dsm(metadata.VisualComplexity, metric='euclidean')

# Plot the DSMs
mne_rsa.plot_dsms([dsm1, dsm2], names=['Word length', 'Vis. complexity'])

###############################################################################
# Perform RSA between the two DSMs using Spearman correlation

rsa_result = mne_rsa.rsa(dsm1, dsm2, metric='spearman')
print('RSA score:', rsa_result)

###############################################################################
# We can compute RSA between multiple DSMs by passing lists to the
# :func:`mne_rsa.rsa` function.

# Create DSMs for each stimulus property
columns = metadata.columns[1:]  # Skip the first column: WORD
dsms = [mne_rsa.compute_dsm(metadata[col], metric='euclidean')
        for col in columns]

# Plot the DSMs
mne_rsa.plot_dsms(dsms, names=columns)

# Compute RSA between the first two DSMs (Concreteness and WordFrequency) and
# the others.
rsa_results = mne_rsa.rsa(dsms[:2], dsms[2:], metric='spearman')

# Pack the result into a Pandas DataFrame for easy viewing
pd.DataFrame(rsa_results, index=columns[:2], columns=columns[2:])

###############################################################################
# What if we have many DSMs? The :func:`mne_rsa.rsa` function is optimized for
# the case where the first parameter (the "data" DSMs) is a large list of DSMs
# and the second parameter (the "model" DSMs) is a smaller list. To save
# memory, you can also pass generators instead of lists.
#
# Let's create a generator that creates DSMs for each time-point in the EEG
# data and compute the RSA between those DSMs and all the "model" DSMs we
# computed above.
#
# This computation will take some time. Therefore, we pass a few extra
# parameters to :func:`mne_rsa.rsa` to enable some improvements. First, the
# ``verbose=True`` enables a progress bar. However, since we are using a
# generator, the progress bar cannot automatically infer how many DSMs there
# will be. Hence, we provide this information explicitly using the
# ``n_data_dsms`` parameter. Finally, depending on how many CPUs you have on
# your system, consider increasing the ``n_jobs`` parameter to parallelize the
# computation over multiple CPUs.

eeg_data = epochs.get_data()
n_trials, n_sensors, n_times = eeg_data.shape


def generate_eeg_dsms():
    for i in range(n_times):
        yield mne_rsa.compute_dsm(eeg_data[:, :, i], metric='correlation')


rsa_results = mne_rsa.rsa(generate_eeg_dsms(), dsms, metric='spearman',
                          verbose=True, n_data_dsms=n_times, n_jobs=1)

# Plot the RSA values over time using standard matplotlib commands
plt.figure(figsize=(8, 4))
plt.plot(epochs.times, rsa_results)
plt.xlabel('time (s)')
plt.ylabel('RSA value')
plt.legend(columns)
