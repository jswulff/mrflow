#! /usr/bin/env python2

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

sourcefiles = ["image.cpp",
                "eff_trws.pyx",
               ]

setup(ext_modules = cythonize(Extension("*",
        sources=sourcefiles,
        include_dirs=[".","./eff_mlabel_ver1"],
        extra_compile_args=["-O3", "-g", "-pg"],
        language="c++",)))
