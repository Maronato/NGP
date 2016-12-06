import numpy
from distutils.core import setup
from Cython.Build import cythonize

# Cython setup file
setup(
    ext_modules = cythonize("helpers.pyx"),
    include_dirs = [numpy.get_include()] #Include directory not hard-wired
)
