from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("voxelize",
              sources=["voxelize.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(
    ext_modules=cythonize(ext_modules)
)
