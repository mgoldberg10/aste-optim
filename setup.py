import setuptools
from distutils.core import setup

setup(
        name        = "asteoptim",
        packages    = ['asteoptim'],
        version     = "0.0",
        author      = "Matthew Goldberg",
        author_email= 'matthew.goldberg10@utexas.edu',
        description = 'ASTE optimization utilities',
        license     = '',
        keywords    = 'MIT License',
        url         = '',
        install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'xarray',
            'xmitgcm',
            ],
        tests_require=['pytest>=2.8']
)
