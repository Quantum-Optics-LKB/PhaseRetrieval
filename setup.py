#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Package meta-data.
NAME = 'ComputeCGH'
DESCRIPTION = 'Python implementation of a holography calculator.'
URL = 'https://github.com/quantumopticslkb/phase_retrieval'
EMAIL = 'tangui.aladjidi@polytechnique.edu'
AUTHOR = 'Tangui Aladjidi'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None

REQUIRED = [
        'scipy',
        'numpy',
        'matplotlib',
	'PIL',
	'LightPipes'
    ]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


setup(
    name=NAME,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(exclude=('tests')),
    install_requires=REQUIRED,
    zip_safe=False,
    include_package_data=True,
)

