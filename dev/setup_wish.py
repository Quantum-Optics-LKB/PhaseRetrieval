from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Cythonized WISH_lkb',
    ext_modules=cythonize("WISH_lkb.pyx"),
    zip_safe=False,
)