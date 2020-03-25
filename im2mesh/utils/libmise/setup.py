from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules = [
    Extension("mise",
              sources=["mise.pyx"],
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)
