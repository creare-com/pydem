# -*- coding: utf-8 -*-
# NOTES
# TODO: Keep track of success in metadata so we don't recompuet
# TODO: Make the edge resolution computation much, much smarter
# TODO: Clean up filenames -- should be attributes of class
# TODO: Make a final seamless result with no overlaps -- option to save out to GeoTIFF, and quantize data for cheaper storage

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
        return (0, fn + ':' + traceback.format_exc())
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
        return (0, fn + ':' + traceback.format_exc())
    return (1, "{}: success".format(fn))

def calc_uca(fn, out_fn_uca, out_fn_todo, out_fn_done, out_fn, out_slice):
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
        save_result(dp.edge_todo.astype(np.bool), out_fn_todo, out_slice, _test_bool)
        save_result(dp.edge_done.astype(np.bool), out_fn_done, out_slice, _test_bool)
    except Exception as e:
        return (0, fn + ':' + traceback.format_exc())
    return (1, "{}: success".format(fn))

EDGE_SLICES = {
        'left': (slice(0, None), 0),
        'right': (slice(0, None), -1),
        'top' : (0, slice(0, None)),
        'bottom' : (-1, slice(0, None))
        }

def calc_uca_ec(fn, out_fn_uca, out_fn_uca_i, out_fn_todo, out_fn_done, out_fn, out_slice, edge_slice):
    #if 1:
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs['elev'] = zarr.open(out_fn, mode='a')['elev'][out_slice]
        kwargs['direction'] = zarr.open(out_fn, mode='r')['aspect'][out_slice]
        kwargs['mag'] = zarr.open(out_fn, mode='a')['slope'][out_slice]
        kwargs['fill_flats'] = False  # assuming we already did this
        dp = DEMProcessor(**kwargs)
        dp.find_flats()

        # get the initial UCA
        uca_init = zarr.open(out_fn_uca, mode='a')[out_slice]
        if np.all(uca_init == 0):
            uca_init = zarr.open(out_fn_uca_i, mode='a')[out_slice]

        # get the edge data
        edge_init_data = {key: zarr.open(out_fn_uca, mode='a')[edge_slice[key]] for key in edge_slice.keys()}
        edge_init_done = {key: zarr.open(out_fn_done, mode='a')[edge_slice[key]] for key in edge_slice.keys()}
        edge_init_todo = {key: zarr.open(out_fn_todo, mode='a')[out_slice][EDGE_SLICES[key]] for key in edge_slice.keys()}

        dp.calc_uca(uca_init=uca_init, edge_init_data=[edge_init_data, edge_init_done, edge_init_todo])
        save_result(dp.uca.astype(np.float32), out_fn_uca, out_slice)
        save_result(dp.edge_todo.astype(np.bool), out_fn_todo, out_slice, _test_bool)
        save_result(dp.edge_done.astype(np.bool), out_fn_done, out_slice, _test_bool)
    except Exception as e:
        return (0, fn + ':' + traceback.format_exc())
    return (1, "{}: success".format(fn))

def _test_float(a, b):
    return np.any(np.abs(a - b) > 1e-8)

def _test_bool(a, b):
    return np.any(a != b)

def save_result(result, out_fn, out_slice, test_func=_test_float):
    # save the results
    outstr = ''
    zf = zarr.open(out_fn, mode='a')
    try:
        zf[out_slice] = result
    except Exception as e:
        outstr += out_fn + '\n' + traceback.format_exc()
    # verify in case another process overwrote my changes
    count = 0
    while test_func(zf[out_slice], result):
        # try again
        try: 
            zf[out_slice] = result
        except Exception as e:
            outstr += ':' + out_fn + '\n' + traceback.format_exc()
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

    def _i(self, name):
        ''' helper function for indexing self.index '''
        return ['left', 'bottom', 'right', 'top', 'dlon', 'dlat', 'nrows', 'ncols'].index(name)

    # Working attributes
    index = tt.Array()  # left, bottom, right, top, dlon, dlat, nrows, ncols 
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
    grid_slice_unique = tl.List()
    grid_slice_noverlap = tl.List()
    grid_shape = tt.Array()
    grid_lat_size = tt.Array()
    grid_lon_size = tt.Array()
    grid_lat_size_unique = tt.Array()
    grid_lon_size_unique = tt.Array()
    grid_size_tot = tt.Array()
    grid_chunk = tt.Array()
    edge_data = tl.List()


    # Properties
    @property 
    def n_inputs(self):
        return len(self.elev_source_files)

    # Methods

    def compute_grid(self):
        # Get the unique lat/lons
        lats = np.round(self.index[:, self._i('top')], self.grid_round_decimals)
        lons = np.round(self.index[:, self._i('left')], self.grid_round_decimals)
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
                grid_lat_size[grid_id[i, 0]] = self.index[i, self._i('nrows')]
            else:
                assert grid_lat_size[grid_id[i, 0]] == self.index[i, self._i('nrows')]

            if grid_lon_size[grid_id[i, 1]] < 0:
                grid_lon_size[grid_id[i, 1]] = self.index[i, self._i('ncols')]
            else:
                assert grid_lon_size[grid_id[i, 1]] == self.index[i, self._i('ncols')]

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

    def _calc_overlap(self, a, da, b, db, s, tie):
        '''
        ...bbbb|bb       db = da/2
           a a |a . . .
        n_overlap_a = 1
        n_overlap_b = 4

        o
        o
        o
        ab
        ab
        ---
        ab
        o
        o
        o
        n_overlap_a = 1
        n_overlap_b = 2
        '''
        n_overlap_a = int(np.round(((b - a) / da + tie - 0.01) / 2))
        n_overlap_a = max(n_overlap_a, 1)  # max with 1 deals with the zero-overlap case
        n_overlap_b = int(np.round(((b - a) / db + 1 - tie - 0.01) / 2))
        n_overlap_b = max(n_overlap_b, 1)  # max with 1 deals with the zero-overlap case
        return n_overlap_a, n_overlap_b

    def compute_grid_overlaps(self):

        self.grid_slice_unique = []
        self.edge_data = []
        for i in range(self.n_inputs):
            id = self.grid_id[i]

            # Figure out the unique slices associated with each file
            slc = self.grid_slice[i]

            # do left-right overlaps
            if id[1] > 0:  # left
                left_id = self.grid_id2i[id[0], id[1] - 1]
                lon_start, lon_start_e = self._calc_overlap(
                        self.index[i, self._i('left')],
                        self.index[i, self._i('dlon')],
                        self.index[left_id, self._i('right')],
                        self.index[left_id, self._i('dlon')],
                        slc[1].start, tie=0)
            else:
                lon_start = lon_start_e = 0
            if id[1] < (self.grid_id2i.shape[1] - 1):  # right
                right_id = self.grid_id2i[id[0], id[1] + 1]
                lon_end, lon_end_e = self._calc_overlap(
                        self.index[right_id, self._i('left')],
                        self.index[i, self._i('dlon')],
                        self.index[i, self._i('right')],
                        self.index[right_id, self._i('dlon')],
                        slc[1].start, tie=1)
            else:
                lon_end = lon_end_e = 0

            # do top-bot overlaps
            if id[0] > 0:  # top
                top_id = self.grid_id2i[id[0] - 1, id[1]]
                lat_start, lat_start_e = self._calc_overlap(
                        self.index[i, self._i('top')],
                        self.index[i, self._i('dlat')],
                        self.index[top_id, self._i('bottom')],
                        self.index[top_id, self._i('dlat')],
                        slc[0].start, tie=0)
            else:
                lat_start = lat_start_e = 0
            if id[0] < (self.grid_id2i.shape[0] - 1):  # bottom
                bot_id = self.grid_id2i[id[0] + 1, id[1]]
                lat_end, lat_end_e = self._calc_overlap(
                        self.index[bot_id, self._i('top')],
                        self.index[i, self._i('dlat')],
                        self.index[i, self._i('bottom')],
                        self.index[bot_id, self._i('dlat')],
                        slc[0].start, tie=1)
            else:
                lat_end = lat_end_e = 0

            self.grid_slice_unique.append((
                slice(slc[0].start + lat_start, slc[0].stop - lat_end),
                slice(slc[1].start + lon_start, slc[1].stop - lon_end)))

            # Figure out where to get edge data from
            edge_data = {
                    'left': (slc[0], slc[1].start - lon_start_e),
                    'right': (slc[0], slc[1].stop + lon_end_e - 1),
                    'top': (slc[0].start - lat_start_e, slc[1]),
                    'bottom': (slc[0].stop + lat_end_e - 1, slc[1]),
                        }
            self.edge_data.append(edge_data)

        # Figure out size of the non-overlapping arrays
        self.grid_lat_size_unique = np.array([
            self.grid_slice_unique[i][0].stop - self.grid_slice_unique[i][0].start 
            for i in self.grid_id2i[:, 0]])
        self.grid_lon_size_unique = np.array([
            self.grid_slice_unique[i][1].stop - self.grid_slice_unique[i][1].start 
            for i in self.grid_id2i[0, :]])

        # Figure out the slices for the non-overlapping arrays
        grid_lat_size_cumulative = np.concatenate([[0], self.grid_lat_size_unique.cumsum()])
        grid_lon_size_cumulative = np.concatenate([[0], self.grid_lon_size_unique.cumsum()])
        grid_slice = []
        for i in range(self.n_inputs):
            ii = self.grid_id[i, 0]
            jj = self.grid_id[i, 1]
            slc = (slice(grid_lat_size_cumulative[ii], grid_lat_size_cumulative[ii + 1]),
                   slice(grid_lon_size_cumulative[jj], grid_lon_size_cumulative[jj + 1]))
            grid_slice.append(slc)

        self.grid_slice_noverlap = grid_slice


    def process_elevation(self, indices=None):
        # TODO: Also find max, mean, and min elevation on edges
        # initialize zarr output
        out_file = os.path.join(self.out_path, 'elev')
        zf = zarr.open(out_file, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a', dtype=np.float32)
        out_file_success = os.path.join(self.out_path, 'success')
        success = zarr.open(out_file_success, shape=(self.n_inputs, 4), mode='a', dtype=bool)

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn=out_file, out_slice=self.grid_slice[i])
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_elev_cond, kwds, success[:, 0])
        self.out_file['success'][:, 0] = success
        return success

    def process_aspect_slope(self):
        # initialize zarr output
        out_aspect = os.path.join(self.out_path, 'aspect')
        zf = zarr.open(out_aspect, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a', dtype=np.float32)
        out_slope = os.path.join(self.out_path, 'slope')
        zf1 = zarr.open(out_slope, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a', dtype=np.float32)

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_aspect=out_aspect,
                     out_fn_slope=out_slope,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i])
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_aspect_slope, kwds, success=self.out_file['success'][:, 1])
        self.out_file['success'][:, 1] = success
        return success

    def process_uca(self):
        # initialize zarr output
        out_uca = os.path.join(self.out_path, 'uca')
        zf = zarr.open(out_uca, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a', dtype=np.float32)
        out_edge_done = os.path.join(self.out_path, 'edge_done')
        zf1 = zarr.open(out_edge_done, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a', dtype=np.bool)
        out_edge_todo = os.path.join(self.out_path, 'edge_todo')
        zf2 = zarr.open(out_edge_todo, shape=self.grid_size_tot, chunk=self.grid_chunk, mode='a', dtype=np.bool)

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_uca=out_uca,
                     out_fn_todo=out_edge_todo,
                     out_fn_done=out_edge_done,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i])
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_uca, kwds, success=self.out_file['success'][:, 2],
                intermediate_fun=self.compute_grid_overlaps)
        self.out_file['success'][:, 2] = success
        return success

    def process_uca_edges(self):
        out_uca_i = os.path.join(self.out_path, 'uca')
        out_uca = os.path.join(self.out_path, 'uca_corrected')
        zf = zarr.open(out_uca, shape=self.grid_size_tot,
                chunk=self.grid_chunk, mode='a', dtype=np.float32,
                fill_value=0)
        out_edge_done = os.path.join(self.out_path, 'edge_done')
        out_edge_todo = os.path.join(self.out_path, 'edge_todo')
        
        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_uca=out_uca,
                     out_fn_uca_i=out_uca_i,
                     out_fn_todo=out_edge_todo,
                     out_fn_done=out_edge_done,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i],
                     edge_slice=self.edge_data[i])
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_uca_ec, kwds, success=self.out_file['success'][:, 3])
        self.out_file['success'][:, 3] = success
        return success

    def queue_processes(self, function, kwds, success=None, intermediate_fun=lambda:None):
        # initialize success if not created
        if success is None:
            success = [0] * len(res)

        # create pool and queue
        pool = Pool(processes=self.n_workers)

        # submit workers
        print ("Sumitting workers", end='...')
        #success = [function(**kwds[i]) 
        #        for i in range(self.n_inputs) if indices[i]]

        res = [pool.apply_async(function, kwds=kwds[i]) 
                for i in range(self.n_inputs) if not success[i]]
        print(" waiting for computation")
        pool.close()  # prevent new tasks from being submitted

        intermediate_fun()

        pool.join()   # wait for tasks to finish

        for i, r in enumerate(res):
            s = r.get(timeout=0.0001)
            success[i] = s[0]
            print(s)
        return success

    def process_twi(self):
        print("Compute Grid")
        self.compute_grid()
        print("Compute Elevation")
        self.process_elevation()
        print("Compute Aspect and Slope")
        self.process_aspect_slope()
        print("Compute UCA")
        self.process_uca()
        print("Compute UCA Corrections")
        self.process_uca_edges()
