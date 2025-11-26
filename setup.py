"""
   Copyright 2015-2024 Creare

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Always perfer setuptools over distutils
from setuptools import setup, Extension
import platform
from os import path
import os
import numpy as np
from Cython.Build import cythonize

here = os.path.join(path.dirname(__file__), 'pydem')

compile_args = ["-O3"]

if '32' in platform.architecture()[0]:
    compile_args.append("-march=i386")
else:
    compile_args.append("-march=x86-64")

# Pattern functions
path_cyfuncs = os.path.join(here, 'cyfuncs')

extensions = [
    Extension("pydem.cyfuncs.cyutils",
              ["pydem/cyfuncs/cyutils.pyx"],
              include_dirs=[np.get_include(), path_cyfuncs],
              library_dirs=[],
              extra_compile_args=compile_args,
              language='c++')
]

setup(
    ext_modules=cythonize(extensions)
)