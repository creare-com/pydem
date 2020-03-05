# -*- coding: utf-8 -*-
import os
import traceback
import subprocess
import numpy as np
import zarr
import scipy.interpolate as spinterp
import gc
import traitlets as tl
import traittypes as tt
import rasterio

from .dem_processing import DEMProcessor
from .utils import parse_fn, sortrows, save_raster, read_raster

class ProcessManager(tl.HasTraits):
    # Parameters controlling the computation
    grid_round_decimals = tl.Int(2)  # Number of decimals to round the lat/lon to for determining the size of the full dataset


    n_workers = tl.Int(1)
    in_path = tl.Unicode('.')
    out_format = tl.Enum(['zarr'], default_value='zarr')
    
    out_path = tl.Unicode()
    @tl.default('out_path')
    def _out_path_default(self):
        out = 'results'
        if self.out_format == 'zarr':
            out += '.zarr'
        return os.path.join(self.in_path, out)
    _INPUT_FILE_TYPES = tl.List(["tif", "tiff", "vrt", "hgt", 'flt', 'adf', 'grib',
                                 'grib2', 'grb', 'gr1', 'zarr'])

    elev_source_files = tl.List()
    @tl.default('elev_source_files')
    def _elev_source_files_default(self):
        esf =  [os.path.join(self.in_path, fn)
                  for fn in os.listdir(self.in_path)
                  if os.path.splitext(fn)[-1].replace('.', '')
                  in self._INPUT_FILE_TYPES]
        esf.sort()
        return esf

    out_file = tl.Any()  # only for .zarr cases
    @tl.default('out_file')
    def _out_file_default(self):
        return zarr.open(self.out_path, mode='a')


    # Working attributes
    index = tt.Array()  # dlon/dj, dlon/di, lonstart, dlat/dj, dlat/di, latstart, nrows, ncols
    @tl.default('index')
    def compute_index(self):
        index = np.zeros((len(self.elev_source_files), 11))
        for i, f in enumerate(self.elev_source_files):
            r = rasterio.open(f, 'r')
            index[i, :9] = np.array(r.transform)
            index[i, 9:] = r.shape
        return index

    grid_id = tt.Array()
    grid_shape = tt.Array()
    grid_size = tt.Array()
    grid_size_tot = tt.Array()
    grid_chunk = tt.Array()
    def compute_grid(self):
        # Get the unique lat/lons
        lats = np.round(self.index[:, 5], self.grid_round_decimals)
        lons = np.round(self.index[:, 2], self.grid_round_decimals)
        ulats = np.sort(np.unique(lats))[::-1].tolist()
        ulons = np.sort(np.unique(lons)).tolist()

        grid_id = np.zeros((self.index.shape[0], 3), dtype=int)
        grid_shape = (len(ulats), len(ulons))
        grid_size = np.ones(grid_shape + (2, ), dtype=int) * -1
        
        # Put the index of filename into the grid, and check that sizes are consistent
        for i in range(self.index.shape[0]):
            grid_id[i, :2] = [ulats.index(lats[i]), ulons.index(lons[i])]
            grid_id[i, 2] = grid_id[i, 1] + grid_id[i, 0] * grid_shape[1]

            # check sizes match
            if grid_size[grid_id[i, 0], grid_id[i, 1], 0] < 0:
                grid_size[grid_id[i, 0], grid_id[i, 1], 0] = self.index[i, 9]
                grid_size[grid_id[i, 0], grid_id[i, 1], 1] = self.index[i, 10]
            else:
                assert grid_size[grid_id[i, 0], grid_id[i, 1], 0] == self.index[i, 9]
                assert grid_size[grid_id[i, 0], grid_id[i, 1], 1] == self.index[i, 10]

        self.grid_id = grid_id
        self.grid_shape = grid_shape
        self.grid_size = grid_size
        self.grid_size_tot = grid_size.sum(axis=(0, 1))
        self.grid_chunk = grid_size.min(axis=(0, 1))


    def process_twi_files(self):
        pass
