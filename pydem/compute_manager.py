# -*- coding: utf-8 -*-
# NOTES
# TODO: Resolve any overlaps between files. Basically no changes for slope/aspect elev,
#       but uca and twi we figure out the unique tiles and construct the whole thing that way
# TODO: Edge resolution and population

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
from multiprocessing import Queue
from multiprocessing import Process
from multiprocessing.pool import Pool

from .dem_processing import DEMProcessor
from .utils import parse_fn, sortrows, save_raster, read_raster, dem_processor_from_raster_kwargs

def calc_elev_cond(fn, out_fn, out_slice):
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        dp = DEMProcessor(**kwargs)
        dp.calc_fill_flats()
        elev = dp.elev.astype(np.float32)
        save_result(elev, out_fn, out_slice)
    except Exception as e:
        return (0, fn + ':' + str(e))
    return (1, "{}: success".format(fn))

def calc_aspect_slope(fn, out_fn_aspect, out_fn_slope, out_fn, out_slice):
    #if 1:
    try: 
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs['elev'] = zarr.open(out_fn, mode='a')['elev'][out_slice]
        kwargs['fill_flats'] = False  # assuming we already did this
        dp = DEMProcessor(**kwargs)
        dp.calc_slopes_directions()
        save_result(dp.direction.astype(np.float32), out_fn_aspect, out_slice)
        save_result(dp.mag.astype(np.float32), out_fn_slope, out_slice)
    except Exception as e:
        return (0, fn + ':' + str(e))
    return (1, "{}: success".format(fn))

def calc_uca(fn, out_fn_uca, out_fn, out_slice):
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs['elev'] = zarr.open(out_fn, mode='a')['elev'][out_slice]
        kwargs['direction'] = zarr.open(out_fn, mode='r')['aspect'][out_slice]
        kwargs['mag'] = zarr.open(out_fn, mode='a')['slope'][out_slice]
        kwargs['fill_flats'] = False  # assuming we already did this
        dp = DEMProcessor(**kwargs)
        dp.find_flats()
        dp.calc_uca()
        save_result(dp.uca.astype(np.float32), out_fn_uca, out_slice)
    except Exception as e:
        return (0, fn + ':' + str(e))
    return (1, "{}: success".format(fn))


def save_result(result, out_fn, out_slice):
    # save the results
    outstr = ''
    zf = zarr.open(out_fn, mode='a')
    try:
        zf[out_slice] = result
    except Exception as e:
        outstr += out_fn + '\n' + str(e)
    # verify in case another process overwrote my changes
    count = 0
    while np.any(np.abs(result - zf[out_slice]) > 1e-8):
        # try again
        try: 
            zf[out_slice] = result
        except Exception as e:
            outstr += ':' + out_fn + '\n' + str(e)
        count += 1
        if count > 5:
            outstr += ':' + out_fn + '\n' + 'OUTPUT IS NOT CORRECTLY WRITTEN TO FILE'
            raise Exception(outstr)

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
                                 'grib2', 'grb', 'gr1'])

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
    index = tt.Array()  # left, bottom, right, top, dlat, dlon, nrows, ncols 
    @tl.default('index')
    def compute_index(self):
        index = np.zeros((len(self.elev_source_files), 8))
        for i, f in enumerate(self.elev_source_files):
            r = rasterio.open(f, 'r')
            index[i, :4] = np.array(r.bounds)
            index[i, 4] = r.transform.a
            index[i, 5] = r.transform.e
            index[i, 6:] = r.shape
        return index

    grid_id = tt.Array()
    grid_id2i = tt.Array()
    grid_slice = tl.List()
    grid_shape = tt.Array()
    grid_lat_size = tt.Array()
    grid_lon_size = tt.Array()
    grid_size_tot = tt.Array()
    grid_chunk = tt.Array()

    # Properties
    @property 
    def n_inputs(self):
        return len(self.elev_source_files)

    # Methods

    def compute_grid(self):
        # Get the unique lat/lons
        lats = np.round(self.index[:, 3], self.grid_round_decimals)
        lons = np.round(self.index[:, 0], self.grid_round_decimals)
        ulats = np.sort(np.unique(lats))[::-1].tolist()
        ulons = np.sort(np.unique(lons)).tolist()

        grid_shape = (len(ulats), len(ulons))
        grid_id = np.zeros((self.n_inputs, 3), dtype=int)
        grid_id2i = -np.ones(grid_shape, dtype=int)
        grid_lat_size = np.ones(len(ulats), dtype=int) * -1
        grid_lon_size = np.ones(len(ulons), dtype=int) * -1
        
        # Put the index of filename into the grid, and check that sizes are consistent
        for i in range(self.n_inputs):
            grid_id[i, :2] = [ulats.index(lats[i]), ulons.index(lons[i])]
            grid_id[i, 2] = grid_id[i, 1] + grid_id[i, 0] * grid_shape[1]
            grid_id2i[grid_id[i, 0], grid_id[i, 1]] = i

            # check sizes match
            if grid_lat_size[grid_id[i, 0]] < 0:
                grid_lat_size[grid_id[i, 0]] = self.index[i, 6]
            else:
                assert grid_lat_size[grid_id[i, 0]] == self.index[i, 6]

            if grid_lon_size[grid_id[i, 1]] < 0:
                grid_lon_size[grid_id[i, 1]] = self.index[i, 7]
            else:
                assert grid_lon_size[grid_id[i, 1]] == self.index[i, 7]

        # Figure out the slice indices in the zarr array for each elevation file
        grid_lat_size_cumulative = np.concatenate([[0], grid_lat_size.cumsum()])
        grid_lon_size_cumulative = np.concatenate([[0], grid_lon_size.cumsum()])
        grid_slice = []
        for i in range(self.n_inputs):
            ii = grid_id[i, 0]
            jj = grid_id[i, 1]
            slc = (slice(grid_lat_size_cumulative[ii], grid_lat_size_cumulative[ii + 1]),
                   slice(grid_lon_size_cumulative[jj], grid_lon_size_cumulative[jj + 1]))
            grid_slice.append(slc)

        self.grid_id = grid_id
        self.grid_id2i = grid_id2i
        self.grid_shape = grid_shape
        self.grid_lat_size = grid_lat_size
        self.grid_lon_size = grid_lon_size
        self.grid_size_tot = [grid_lat_size.sum(), grid_lon_size.sum()]
        self.grid_slice = grid_slice
        self.grid_chunk = [grid_lat_size.min(), grid_lon_size.min()]

    def compute_grid_overlaps(self):
        # Figure out the unique slices associated with each file
        self.grid_slice_unique = None

        # Figure out where to get new data on edges for each uca, this is at the start of a new round
        self.edge_data = [{'left': slice, 'right': slice, 'top': slice, 'bottom': slice}]

        # Figure out neighbors, after computing updates, will post that these neighbors have new data
        self.neighbors =  [{'left': id, 'right': id, 'top': id, 'bottom': id}]

    def process_elevation(self, indices=None):
        # TODO: Also find max, mean, and min elevation on edges
        # initialize zarr output
        out_file = os.path.join(self.out_path, 'elev')
        zf = zarr.open(out_file, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a')

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn=out_file, out_slice=self.grid_slice[i])
                for i in range(self.n_inputs)]

        # TODO: Save this to prevent recomputation
        success = self.queue_processes(calc_elev_cond, kwds)
        return success

    def process_aspect_slope(self):
        # initialize zarr output
        out_aspect = os.path.join(self.out_path, 'aspect')
        zf = zarr.open(out_aspect, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a')
        out_slope = os.path.join(self.out_path, 'slope')
        zf = zarr.open(out_slope, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a')

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_aspect=out_aspect,
                     out_fn_slope=out_slope,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i])
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_aspect_slope, kwds)
        return success

    def process_uca(self):
        # initialize zarr output
        out_uca = os.path.join(self.out_path, 'uca')
        zf = zarr.open(out_uca, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a')
        # Also initialize done and todo, on the overlap space

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_uca=out_uca,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i])
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_uca, kwds)
        return success

    def process_uca_edges(self):
        pass

    def queue_processes(self, function, kwds, indices=None):
        # initialize indices if not created
        if indices is None:
            indices = np.ones(self.n_inputs, bool)

        # create pool and queue
        pool = Pool(processes=self.n_workers)

        # submit workers
        print ("Sumitting workers", end='...')
        #success = [function(**kwds[i]) 
        #        for i in range(self.n_inputs) if indices[i]]
        res = [pool.apply_async(function, kwds=kwds[i]) 
                for i in range(self.n_inputs) if indices[i]]
        print(" waiting for computation")
        pool.close()  # prevent new tasks from being submitted
        pool.join()   # wait for tasks to finish
        
        success = [r.get(timeout=0.0001) for r in res]
        print('\n'.join([s[1] for s in success]))
        return success

    def process_twi(self):
        self.compute_grid()
        self.process_elevation()
        self.process_aspect_slope()
        self.process_uca()
