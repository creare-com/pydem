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
from setuptools import setup, find_packages, Extension
from codecs import open  # To use a consistent encoding
import platform

from os.path import sep
from os import path
import os
import numpy as np

from Cython.Build import cythonize

here = os.path.join(path.dirname(__file__), 'pydem')

compile_args = []
compile_args.append("-O3")

if '32' in platform.architecture()[0]:
    compile_args.append("-march=i386")
else:
    compile_args.append("-march=x86-64")

# Pattern functions
path_cyfuncs = os.path.join(here, 'cyfuncs')
path_reader = os.path.join(here, 'reader')

extensions = [
    Extension("pydem.cyfuncs.cyutils",
              [os.path.join(path_cyfuncs, "cyutils.pyx")],
              include_dirs=[np.get_include(), path_cyfuncs],
              library_dirs=[],
              extra_compile_args=compile_args,
              language='c++'),
]

setup(
    ext_modules=cythonize(extensions),

    name='pyDEM',

    version='1.1.0',

    description="Software for calculating Topographic Wetness Index (TWI)",
    author='MPU, RXC, JXM',
    url="https://github.com/creare-com/pydem",

    license="APACHE 2.0",

    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: Apache Software License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),

    python_requires='>=3',

    install_requires=[
        'rasterio',
        'numpy',
        'scipy',
        'geopy',
        'traitlets',
        'traittypes',
        'zarr',
        'cython'
        ],

    entry_points = {
        'console_scripts' : ['TWIDinf=pydem.commandline_utils:TWIDinf',
                             'AreaDinf=pydem.commandline_utils:AreaDinf',
                             'DinfFlowDir=pydem.commandline_utils:DinfFlowDir']
    }

)
