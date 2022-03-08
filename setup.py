#! /usr/bin/env python
from setuptools import setup
import os
import os.path as op
import codecs

# get the version (don't import mne-rsa here, so dependencies are not needed)
version = None
with open(op.join('mne_rsa', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    with codecs.open('README.rst', 'r', encoding='utf8') as fid:
        long_description = fid.read()

    setup(
        name='mne-rsa',
        maintainer='Marijn van Vliet',
        maintainer_email='w.m.vanvliet@gmail.com',
        description='Code for performing Representational Similarity Analysis on MNE-Python data structures.',
        long_description_content_type='text/x-rst',
        license='BSD-3',
        url='https://github.com/wmvanvliet/mne-rsa',
        version=version,
        download_url='https://github.com/wmvanvliet/mne-rsa/archive/master.zip',
        long_description=long_description,
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
        install_requires=['numpy', 'scipy', 'scikit-learn', 'nibabel',
                          'matplotlib', 'mne'],
    )
