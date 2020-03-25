from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules = [
    Extension("simplify_mesh",
              sources=["simplify_mesh.pyx"]
              )
]

setup(
    ext_modules=cythonize(ext_modules)
)
