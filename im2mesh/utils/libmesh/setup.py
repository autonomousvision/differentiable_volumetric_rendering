from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules = [
    Extension("triangle_hash",
              sources=["triangle_hash.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(
    ext_modules=cythonize(ext_modules)
)
