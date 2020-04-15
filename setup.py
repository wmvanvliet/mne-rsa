#! /usr/bin/env python
from setuptools import setup
import os

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name='mne-rsa',
        maintainer='Marijn van Vliet',
        maintainer_email='w.m.vanvliet@gmail.com',
        description='Code for performing Representational Similarity Analysis on MNE-Python data structures.',
        license='BSD-3',
        url='https://github.com/wmvanvliet/mne-rsa',
        version='0.4.dev0',
        download_url='https://github.com/wmvanvliet/mne-rsa/archive/master.zip',
        long_description=open('README.md').read(),
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Operating System :: Microsoft :: Windows',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
        platforms='any',
        packages=['mne_rsa'],
        install_requires=['numpy', 'scipy', 'matplotlib', 'mne'],
    )
