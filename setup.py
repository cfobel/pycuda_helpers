#!/usr/bin/env python

from distutils.core import setup

setup(name = "pycuda_helpers",
    version = "0.0.1",
    description = "pycuda helper functions and classes",
    keywords = "pycuda helper",
    author = "Christian Fobel",
    url = "https://github.com/cfobel/pycuda_helpers",
    license = "GPL",
    long_description = """""",
    packages = ['pycuda_helpers', 'pycuda_helpers.shared_storage'],
    package_data={'pycuda_helpers': ['pycuda_include/*'],
                  'pycuda_helpers.shared_storage': ['pycuda_templates/*',
                          'pycuda_include/*']}
)
