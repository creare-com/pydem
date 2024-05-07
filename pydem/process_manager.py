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

# -*- coding: utf-8 -*-
import os
import traceback
import subprocess
import time
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
from multiprocessing.context import TimeoutError as mpTimeoutError
# import ipydb

from .dem_processing import DEMProcessor
from .utils import dem_processor_from_raster_kwargs

EDGE_SLICES = {
        'left': (slice(0, None), 0),
        'right': (slice(0, None), -1),
        'top' : (0, slice(0, None)),
        'bottom' : (-1, slice(0, None)),
        'top-left': (0, 0),
        'top-right' : (0, -1),
        'bottom-right': (-1, -1),
        'bottom-left' : (-1, 0),
        }

DEBUG = False

def calc_elev_cond(fn, out_fn, out_slice, dtype=np.float64, dp_kwargs={}):
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs.update(dp_kwargs)
        dp = DEMProcessor(**kwargs)
        if DEBUG:
            dp.dX[:] = 1
            dp.dY[:] = 1
            dp.dX2[:] = 1
            dp.dY2[:] = 1
        dp.calc_fill_flats()
        dp.calc_pit_drain_paths()

        elev = dp.elev.astype(dtype)
        save_result(elev, out_fn, out_slice)
    except Exception as e:
        return (0, fn + ':' + traceback.format_exc())
    return (1, "{}: success".format(fn))

def calc_aspect_slope(fn, out_fn_aspect, out_fn_slope, out_fn, out_slice, dtype=np.float64, dp_kwargs={}):
    #if 1:
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs['elev'] = zarr.open(out_fn, mode='a')['elev'][out_slice]
        kwargs['fill_flats'] = False  # assuming we already did this
        kwargs.update(dp_kwargs)
        dp = DEMProcessor(**kwargs)
        if DEBUG:
            dp.dX[:] = 1
            dp.dY[:] = 1
            dp.dX2[:] = 1
            dp.dY2[:] = 1

        dp.calc_slopes_directions()
        save_result(dp.direction.astype(dtype), out_fn_aspect, out_slice)
        save_result(dp.mag.astype(dtype), out_fn_slope, out_slice)
    except Exception as e:
        return (0, fn + ':' + traceback.format_exc())
    return (1, "{}: success".format(fn))

def calc_uca(fn, out_fn_uca, out_fn_todo, out_fn_done, out_fn, out_slice, edge_slice, dtype=np.float64, dp_kwargs={}):
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs['elev'] = zarr.open(out_fn, mode='a')['elev'][out_slice]
        kwargs['direction'] = zarr.open(out_fn, mode='r')['aspect'][out_slice]
        kwargs['mag'] = zarr.open(out_fn, mode='a')['slope'][out_slice]
        kwargs['fill_flats'] = False  # assuming we already did this

        downstream_edges = {
            'left': [np.pi / 2, 3 * np.pi / 2],
            'right': [2 * np.pi - np.pi / 2, np.pi / 2],
            'top': [0, np.pi],
            'bottom': [np.pi, 2*np.pi]
            }

        # Fix the aspect and mag for overlap1 case
        for key in ['left', 'right', 'top', 'bottom']:
            if not check_1overlap(out_slice, edge_slice[key]):
                continue
            # only replace downstream edges
            d = kwargs['direction'][EDGE_SLICES[key]]
            if key == 'right':
                ids = ((d >= downstream_edges[key][0]) | (d <= downstream_edges[key][1])) & (d >= 0)
            else:
                ids = (d >= downstream_edges[key][0]) & (d <= downstream_edges[key][1])
            if key in ['top', 'bottom']:
                ids = ids | ((d < 1e-6) & (d >= 0)) | (np.abs(d - np.pi * 2) < 1e-6)
            # also replace any flats
            ids = ids | (d == -1)
            # Don't do corners, sometimes -- these are handled explicitly by the next block
            if key in ['left', 'right']:
                ids[0] = ids[0] & (not ((d[0] >= downstream_edges['top'][0]) & (d[0] <= downstream_edges['top'][1])))
                ids[-1] = ids[-1] & (not ((d[-1] >= downstream_edges['bottom'][0]) & (d[-1] <= downstream_edges['bottom'][1])))
            else:
                ids[0] = ids[0] & (not ((d[0] >= downstream_edges['left'][0]) & (d[0] <= downstream_edges['left'][1])))
                ids[-1] = ids[-1] & (not ((d[-1] >= downstream_edges['right'][0]) | (d[-1] <= downstream_edges['right'][1])) & (d[-1] >= 0))

            kwargs['direction'][EDGE_SLICES[key]][ids] = zarr.open(out_fn, mode='r')['aspect'][edge_slice[key]][ids]
            kwargs['mag'][EDGE_SLICES[key]][ids] = zarr.open(out_fn, mode='r')['slope'][edge_slice[key]][ids]

            # Also update the direction in the output file
            if key == 'left':
                slc = (out_slice[0], out_slice[1].start)
            elif key == 'right':
                slc = (out_slice[0], out_slice[1].stop - 1)
            elif key == 'top':
                slc = (out_slice[0].start, out_slice[1])
            elif key == 'bottom':
                slc = (out_slice[0].stop - 1, out_slice[1])

            zarr.open(out_fn, mode='a')['aspect'][slc] = kwargs['direction'][EDGE_SLICES[key]]
            zarr.open(out_fn, mode='a')['slope'][slc] = kwargs['mag'][EDGE_SLICES[key]]

        for key in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
            if not check_1overlap(out_slice, edge_slice[key]):
                continue
            keytb, keylr = key.split('-')
            # only replace downstream corners
            d = kwargs['direction'][EDGE_SLICES[key]]
            if keylr == 'right':
                ids = (((d >= downstream_edges[keylr][0]) | (d <= downstream_edges[keylr][1])) & (d >= 0)) \
                    & ((d >= downstream_edges[keytb][0]) & (d <= downstream_edges[keytb][1]) | \
                           (((d < 1e-6) & (d >= 0)) | (np.abs(d - np.pi * 2) < 1e-6))
                       )
            else:
                ids = (d >= downstream_edges[keylr][0]) & (d <= downstream_edges[keylr][1]) \
                    & ((d >= downstream_edges[keytb][0]) & (d <= downstream_edges[keytb][1]) | \
                           (((d < 1e-6) & (d >= 0)) | (np.abs(d - np.pi * 2) < 1e-6))
                       )
            #ids = ids | d == -1
            if not ids:
                continue
            slc = edge_slice[key]
            kwargs['direction'][EDGE_SLICES[key]] = zarr.open(out_fn, mode='r')['aspect'][slc]
            kwargs['mag'][EDGE_SLICES[key]] = zarr.open(out_fn, mode='r')['slope'][slc]

            # Also update the direction in the output file
            if key == 'top-left':
                slc = (out_slice[0].start, out_slice[1].start)
            elif key == 'top-right':
                slc = (out_slice[0].start, out_slice[1].stop - 1)
            elif key == 'bottom-left':
                slc = (out_slice[0].stop - 1, out_slice[1].start)
            elif key == 'bottom-right':
                slc = (out_slice[0].stop - 1, out_slice[1].stop - 1)
            zarr.open(out_fn, mode='a')['aspect'][slc] = kwargs['direction'][EDGE_SLICES[key]]
            zarr.open(out_fn, mode='a')['slope'][slc] = kwargs['mag'][EDGE_SLICES[key]]

        kwargs.update(dp_kwargs)
        dp = DEMProcessor(**kwargs)
        if DEBUG:
            dp.dX[:] = 1
            dp.dY[:] = 1
            dp.dX2[:] = 1
            dp.dY2[:] = 1

        dp.find_flats()
        dp.calc_uca(plotflag=False)
        save_result(dp.uca.astype(dtype), out_fn_uca, out_slice)
        save_result(dp.edge_todo.astype(bool), out_fn_todo, out_slice, _test_bool)
        save_result(dp.edge_done.astype(bool), out_fn_done, out_slice, _test_bool)
    except Exception as e:
        return (0, fn + ':' + traceback.format_exc())
    return (1, "{}: success".format(fn))

def calc_uca_ec_metrics(fn_uca, fn_done, fn_todo, slc, edge_slc):
        # get the edge data
    try:

        edge_data = [zarr.open(fn_uca, mode='a')[edge_slc[key]] for key in edge_slc.keys()]
        edge_done = [zarr.open(fn_done, mode='a')[edge_slc[key]]
                for key in edge_slc.keys()]
        edge_todo = [zarr.open(fn_todo, mode='a')[slc][EDGE_SLICES[key]]
                for key in edge_slc.keys()]
        # n_coulddo = 0
        p_done = 0
        n_done = 0
        for ed, et, edn in zip(edge_data, edge_todo, edge_done):
            # coulddo_ = et & (ed > 0)  # previously
            coulddo = et & edn
            n_done += (coulddo).sum()
            p_done += et.sum()
            #n_coulddo += coulddo_.sum()
        p_done = n_done / (1e-16 + p_done)
        # return (n_coulddo, p_done, n_done)
        return [(p_done, n_done)]
    except:
        return [(None, None)]


def calc_uca_ec(fn, out_fn_uca, out_fn_uca_edges, out_fn_todo, out_fn_done, out_fn, out_slice, edge_slice, dtype=np.float64, dp_kwargs={}):
    #if 1:
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs['elev'] = zarr.open(out_fn, mode='a')['elev'][out_slice]
        kwargs['direction'] = zarr.open(out_fn, mode='r')['aspect'][out_slice]
        kwargs['mag'] = zarr.open(out_fn, mode='a')['slope'][out_slice]
        kwargs['fill_flats'] = False  # assuming we already did this
        kwargs.update(dp_kwargs)
        dp = DEMProcessor(**kwargs)
        if DEBUG:
            dp.dX[:] = 1
            dp.dY[:] = 1
            dp.dX2[:] = 1
            dp.dY2[:] = 1

        dp.find_flats()

        # get the initial UCA
        uca_file = zarr.open(out_fn_uca, mode='a')
        uca_init = uca_file[out_slice]
        uca_edges_file = zarr.open(out_fn_uca_edges, mode='a')
        uca_edges_init = uca_edges_file[out_slice]
        done_file = zarr.open(out_fn_done, mode='a')
        todo_file = zarr.open(out_fn_todo, mode='a')

        # get the edge data for the sides
        keys = ['left', 'right', 'top', 'bottom']
        edge_init_data = {key: uca_file[edge_slice[key]] + uca_edges_file[edge_slice[key]] for key in keys}
        edge_init_done = {key: done_file[edge_slice[key]] for key in keys}
        edge_init_todo = {key: todo_file[out_slice][EDGE_SLICES[key]] for key in keys}
        edge_init_todo_neighbor = {key: todo_file[edge_slice[key]] for key in keys}

        # Add in the data for the corners -- for the single-pixel overlap case
        keys = ['top-left', 'bottom-right', 'top-right', 'bottom-left']
        for key in keys:
            # Note, the dem_processor doesn't know about incoming corners, so all changes have to happen on edges
            keytb, keylr = key.split('-')
            inds = EDGE_SLICES[key]
            overlap = edge_init_done[keytb][inds[1]] & edge_init_done[keylr][inds[0]]
            # We will be double-dipping on the corner, so just remove one of the edges from consideration
            if overlap:
                edge_init_done[keytb][inds[1]] = False
                # When I have two choices of edges, the safest choice is actually the real corner, so select that...
                if check_1overlap(out_slice, edge_slice[key]) and done_file[edge_slice[key]]:
                    edge_init_data[keylr][inds[0]] = uca_file[edge_slice[key]] + uca_edges_file[edge_slice[key]]
                    edge_init_data[keytb][inds[1]] = edge_init_data[keylr][inds[0]]

        # fix my TODO if my neighbor has TODO on the same edge -- that should never happen except for floating point
        # rounding errors
        edge_init_todo = {k: v & (edge_init_todo_neighbor[k] == False) for k, v in edge_init_todo.items()}
        #dp.calc_uca(plotflag=True)
        uca = dp.calc_uca(uca_init=uca_init + uca_edges_init,
                          edge_init_data=[edge_init_data, edge_init_done, edge_init_todo],
                          plotflag=False)
        save_result((dp.uca - uca_init).astype(dtype), out_fn_uca_edges, out_slice)
        save_result(dp.edge_todo.astype(bool), out_fn_todo, out_slice, _test_bool)
        save_result(dp.edge_done.astype(bool), out_fn_done, out_slice, _test_bool)
    except Exception as e:
        return (0, fn + ':' + traceback.format_exc(), None)
    return (1, "{}: success".format(fn), uca)

def check_1overlap(out_slice, edge_slc):
    edge_slc = [getattr(e, 'start', e) for e in edge_slc]
    e_c = np.array(edge_slc)
    e_c[0] = np.clip(e_c[0], out_slice[0].start, out_slice[0].stop - 1)
    e_c[1] = np.clip(e_c[1], out_slice[1].start, out_slice[1].stop - 1)
    if any(np.abs(e_c - edge_slc) == 1):
        return True
    return False


def calc_twi(fn, out_fn_twi, out_fn, out_slice, dtype=np.float64, dp_kwargs={}):
    try:
        kwargs = dem_processor_from_raster_kwargs(fn)
        kwargs['elev'] = zarr.open(out_fn, mode='a')['elev'][out_slice]
        kwargs['direction'] = zarr.open(out_fn, mode='r')['aspect'][out_slice]
        kwargs['mag'] = zarr.open(out_fn, mode='a')['slope'][out_slice]
        kwargs['fill_flats'] = False  # assuming we already did this
        kwargs['uca'] = zarr.open(out_fn, mode='a')['uca'][out_slice] + zarr.open(out_fn, mode='a')['uca_edges'][out_slice]
        kwargs.update(dp_kwargs)
        dp = DEMProcessor(**kwargs)
        if DEBUG:
            dp.dX[:] = 1
            dp.dY[:] = 1

        dp.find_flats()
        dp.calc_twi()
        save_result(dp.twi.astype(dtype), out_fn_twi, out_slice)
    except Exception as e:
        return (0, fn + ':' + traceback.format_exc())
    return (1, "{}: success".format(fn))

def calc_overview(fn, out_fn, factor, in_slice, out_slice):
    try:
        outfile = zarr.open(out_fn, mode='a')
        data = zarr.open(fn, mode='r')[in_slice]

        if np.all(data == 0):
            return (0, 'all zero')

        # do calculation
        easy_slice = tuple([slice(0, (s // factor) * factor) for s in data.shape])
        overview = np.zeros([int(np.ceil(s / factor)) for s in data.shape])
        eo_slice = tuple([slice(0, (s // factor)) for s in data.shape])
        easy = data[easy_slice].reshape(data.shape[0] // factor, factor, data.shape[1] // factor, factor)
        easy = np.moveaxis(easy, 1, -2).reshape(data.shape[0] // factor, data.shape[1] // factor, factor ** 2)
        overview[eo_slice] = easy.mean(axis=-1)

        # do edges
        # right edge
        right_edge = data[easy_slice[0], easy_slice[1].stop:].reshape(data.shape[0] // factor, -1)
        right_edge = right_edge.mean(axis=1)
        overview[eo_slice[0], eo_slice[1].stop:] = right_edge[:, None]

        # Bottom edge
        bottom_edge = data[easy_slice[0].stop:, easy_slice[1]].reshape(-1, data.shape[1] // factor)
        bottom_edge = bottom_edge.mean(axis=0)
        overview[eo_slice[0].stop:, eo_slice[1]] = bottom_edge[None, :]

        # corner
        corner_edge = data[easy_slice[0].stop:, easy_slice[1].stop:].mean()
        overview[eo_slice[0].stop:, eo_slice[1].stop:] = corner_edge

        # save results
        outfile[out_slice] = overview
    except Exception:
        return (0, traceback.format_exc())

    return (1, (in_slice, factor))


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

def copy_result(zf1, zf2, slc1, slc2, zf3=None):
    zf1 = zarr.open(zf1, 'a')
    zf2 = zarr.open(zf2, 'r')
    zf1[slc1] = zf2[slc2]
    if zf3 is not None:
        zf3 = zarr.open(zf3, 'r')
        zf1[slc1] += zf3[slc2]
    return (1, 1)


class ProcessManager(tl.HasTraits):
    _debug = tl.Bool(False)
    _true_uca = tt.Array(allow_none=True)
    _true_uca_overlap = tt.Array(allow_none=True)

    dtype = np.float64

    # DEM processes key-word arguments
    dem_proc_kwargs = tl.Dict(key_trait=tl.Enum(
        [
            "fill_flats",
            "fill_flats_below_sea",
            "fill_flats_source_tol",
            "fill_flats_peaks",
            "fill_flats_pits",
            "fill_flats_max_iter",
            "drain_pits",
            "drain_pits_path",
            "drain_pits_min_border",
            "drain_pits_spill",
            "drain_flats",
            "drain_pits_max_iter",
            "drain_pits_max_dist",
            "drain_pits_max_dist_XY",
            # When resolving drainage across edges, if maximum UCA is reached, should
            # edge be marked as completed?
            "apply_uca_limit_edges",
            # When calculating TWI, should TWI be limited to max value?
            "apply_twi_limits",
            # When calculating TWI, should UCA be limited to max value?
            "apply_twi_limits_on_uca",
            "uca_saturation_limit",
            "twi_min_slope",
            "twi_min_area",
            # Mostly deprecated, but maximum number of iterations used to try and
            # resolve circular drainage patterns (which should never occur)
            "circular_ref_maxcount",
            # Default maximum surface area of pit artifacts resulting from quantization
            # that will be filled
            "maximum_pit_area",
    ]))

    @tl.default('_true_uca_overlap')
    def _true_uca_overlap_default(self):
        _true_uca_overlap = np.zeros(self.grid_size_tot) * np.nan
        for i in range(self.n_inputs):
            _true_uca_overlap[self.grid_slice_unique[i]] = self._true_uca[self.grid_slice_noverlap[i]]
        return _true_uca_overlap

    # Parameters controlling the computation
    grid_round_decimals = tl.Int(2)  # Number of decimals to round the lat/lon to for determining the size of the full dataset


    n_workers = tl.Int(1)
    in_path = tl.Unicode('.')
    out_format = tl.Enum(['zarr'], default_value='zarr')

    out_path_noverlap = tl.Unicode()
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

    out_file_noverlap = tl.Any()  # only for .zarr cases
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
    grid_shape = tl.List()
    grid_lat_size = tt.Array()
    grid_lon_size = tt.Array()
    grid_lat_size_unique = tt.Array()
    grid_lon_size_unique = tt.Array()
    grid_size_tot = tl.List()
    grid_size_tot_unique = tl.List()
    grid_chunk = tl.List()
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
        grid_lat_size_cumulative = np.concatenate([[0], grid_lat_size.cumsum().astype(int)])
        grid_lon_size_cumulative = np.concatenate([[0], grid_lon_size.cumsum().astype(int)])
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
        self.grid_size_tot = [int(grid_lat_size.sum()), int(grid_lon_size.sum())]
        self.grid_slice = grid_slice
        self.grid_chunk = [int(grid_lat_size.min()), int(grid_lon_size.min())]

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


           |b b b b
        n n o n n n
        a a a|

        a a a b b b b
        '''
        n_overlap_a = int(np.round(((b - a) / da + tie - 0.01) / 2))
        #n_overlap_a = max(n_overlap_a, 1)  # max with 1 deals with the zero-overlap case
        n_overlap_b = int(np.round((b - a) / db))
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
                if left_id < 0:
                    lon_start = lon_start_e = 0
                else:
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
                if right_id < 0:
                    lon_end = lon_end_e = 0
                else:
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
                if top_id < 0:
                    lat_start = lat_start_e = 0
                else:
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
                if bot_id < 0:
                    lat_end = lat_end_e = 0
                else:
                    lat_end, lat_end_e = self._calc_overlap(
                        self.index[bot_id, self._i('top')],
                        self.index[i, self._i('dlat')],
                        self.index[i, self._i('bottom')],
                        self.index[bot_id, self._i('dlat')],
                        slc[0].start, tie=1)
            else:
                lat_end = lat_end_e = 0

            # Do corner overlaps
            # top left
            if id[1] > 0 and id[0] > 0:
                corner_tl = 1
            else:
                corner_tl = 0
            # bottom left
            if id[1] > 0 and id[0] < (self.grid_id2i.shape[0] - 1):
                corner_bl = 1
            else:
                corner_bl = 0
            # top right
            if id[1] < (self.grid_id2i.shape[1] - 1) and id[0] > 0:
                corner_tr = 1
            else:
                corner_tr = 0
            # bottom right
            if id[1] < (self.grid_id2i.shape[1] - 1) and id[0] < (self.grid_id2i.shape[0] - 1):
                corner_br = 1
            else:
                corner_br = 0

            self.grid_slice_unique.append((
                slice(slc[0].start + lat_start, slc[0].stop - lat_end),
                slice(slc[1].start + lon_start, slc[1].stop - lon_end)))

            # Figure out where to get edge data from
            edge_data = {
                    'left': (slc[0], slc[1].start - lon_start_e),
                    'right': (slc[0], slc[1].stop + lon_end_e - 1),
                    'top': (slc[0].start - lat_start_e, slc[1]),
                    'top-left': (slc[0].start - lat_start_e * corner_tl, slc[1].start - lon_start_e * corner_tl),
                    'top-right': (slc[0].start - lat_start_e  * corner_tr, slc[1].stop + lon_end_e * corner_tr - 1),
                    'bottom': (slc[0].stop + lat_end_e - 1, slc[1]),
                    'bottom-left': (slc[0].stop + lat_end_e * corner_bl - 1, slc[1].start - lon_start_e * corner_bl),
                    'bottom-right': (slc[0].stop + lat_end_e * corner_br - 1, slc[1].stop + lon_end_e * corner_br - 1),
                        }
            self.edge_data.append(edge_data)

        # Figure out size of the non-overlapping arrays
        self.grid_lat_size_unique =  np.nanmin(np.array(
            [
                [
                    self.grid_slice_unique[i][0].stop - self.grid_slice_unique[i][0].start + 0 / np.array(i != -1)
                    for i in self.grid_id2i[:, j]
                ]
                for j in range(self.grid_id2i.shape[1])
            ]), axis=0)
        self.grid_lon_size_unique = np.nanmin(np.array(
            [
                [
                    self.grid_slice_unique[i][1].stop - self.grid_slice_unique[i][1].start + 0 / np.array(i != -1)
                    for i in self.grid_id2i[j, :]
                ]
                for j in range(self.grid_id2i.shape[0])
            ]), axis=0)

        # Figure out the slices for the non-overlapping arrays
        grid_lat_size_cumulative = np.concatenate([[0], self.grid_lat_size_unique.cumsum().astype(int)])
        grid_lon_size_cumulative = np.concatenate([[0], self.grid_lon_size_unique.cumsum().astype(int)])
        grid_slice = []
        for i in range(self.n_inputs):
            ii = self.grid_id[i, 0]
            jj = self.grid_id[i, 1]
            slc = (slice(grid_lat_size_cumulative[ii], grid_lat_size_cumulative[ii] \
                         + self.grid_slice_unique[i][0].stop - self.grid_slice_unique[i][0].start),
                   slice(grid_lon_size_cumulative[jj], grid_lon_size_cumulative[jj] \
                         + self.grid_slice_unique[i][1].stop - self.grid_slice_unique[i][1].start))
            grid_slice.append(slc)

        self.grid_slice_noverlap = grid_slice
        self.grid_size_tot_unique = [int(self.grid_lat_size_unique.sum()),
                                     int(self.grid_lon_size_unique.sum())]

    def save_non_overlap_data(self, new_fn=None, keys=['elev', 'uca', 'aspect', 'slope', 'twi'], chunks=None):
        if new_fn is None:
            new_fn = self.out_path.replace('.zarr', '') + '_compact.zarr'

        if chunks is None:
            chunks = [int(self.grid_size_tot_unique[0] // self.grid_lat_size_unique.size),
                      int(self.grid_size_tot_unique[1] // self.grid_lon_size_unique.size)]


        if self.n_workers == 1:
            for key in keys:
                zf = zarr.open(os.path.join(new_fn, key),
                               shape=self.grid_size_tot_unique,
                               chunks=chunks, mode='a', dtype=self.dtype)
                for i in range(self.n_inputs):
                    zf[self.grid_slice_noverlap[i]] = self.out_file[key][self.grid_slice_unique[i]]
                    if key == 'uca':
                        zf[self.grid_slice_noverlap[i]] += self.out_file[key + '_edges'][self.grid_slice_unique[i]]

            self.out_file_noverlap = zarr.open(new_fn)
            self.out_path_noverlap = new_fn
            return

        # populate kwds
        kwds = []
        for key in keys:
            entry = {}
            entry['zf1'] = os.path.join(new_fn, key)
            # Initialize file
            zf = zarr.open(entry['zf1'],
                           shape=self.grid_size_tot_unique,
                           chunks=chunks, mode='a', dtype=self.dtype)
            entry['zf2'] = os.path.join(self.out_path, key)
            if key == 'uca':
                entry['zf3'] = os.path.join(self.out_path, key + '_edges')
            for i in range(self.n_inputs):
                e = entry.copy()
                e['slc1'] = self.grid_slice_noverlap[i]
                e['slc2'] = self.grid_slice_unique[i]
                kwds.append(e)

        success = self.queue_processes(copy_result, kwds)
        return success

    def save_non_overlap_data_geotiff(
            self,
            dtype,
            crs=None,
            new_path=None,
            keys=['elev', 'uca', 'aspect', 'slope', 'twi'],
            chunks=None,
            overview_type=None,
            overview_factors=None,
            rescale=None):
        if new_path is None:
            new_path = self.out_path.replace('.zarr', '')

        if chunks is None:
            chunks = [512, 512]

        if crs is None:
            crs = rasterio.open(self.elev_source_files[0], 'r').crs

        # Create the geotransform(s) for the large file
        decimals_to_round = 14
        lat_check_ids = self.grid_id2i.max(axis=1)
        dlats = np.unique(np.round(self.index[lat_check_ids, self._i('dlat')], decimals = decimals_to_round))
        lon_check_ids = self.grid_id2i.max(axis=0)
        dlons = np.unique(np.round(self.index[lon_check_ids, self._i('dlon')], decimals = decimals_to_round))
        if (dlats.size > 1) or (dlons.size > 1):
            raise NotImplementedError

        top = self.index[:, self._i('top')].max()
        bottom = self.index[:, self._i('bottom')].min()
        left = self.index[:, self._i('left')].min()
        right = self.index[:, self._i('right')].max()
        dlat = (bottom - top) / self.grid_size_tot_unique[0]
        dlon = (right - left) / self.grid_size_tot_unique[1]
        transform = rasterio.transform.Affine.translation(left, top) * rasterio.transform.Affine.scale(dlon, dlat)
        for key in keys:
            filename = os.path.join(new_path, key + '.tiff')
            with rasterio.open(
                    filename, 'w', compress="LZW",
                    width=self.grid_size_tot_unique[1], height=self.grid_size_tot_unique[0],
                    tiled=True, blockxsize=chunks[0], blockysize=chunks[1],
                    count=1,
                    dtype=dtype,
                    driver='GTiff',
                    bigtiff=True,
                    crs=crs,
                    transform=transform
                    ) as dataset:

                for i in range(self.n_inputs):
                    print ("Writing {} of {}".format(i, self.n_inputs))
                    data = self.out_file[key][self.grid_slice_unique[i]]
                    if key == 'uca':
                        data += self.out_file[key + '_edges'][self.grid_slice_unique[i]]
                    slc = self.grid_slice_noverlap[i]
                    window = rasterio.windows.Window.from_slices(*slc)
                    if rescale:
                        data = (data - rescale[0]) / (rescale[1] - rescale[0]) * rescale[2]

                    dataset.write(data.astype(dtype), window=window, indexes=1)

                if overview_type is not None:
                    if overview_factors is None:
                        overview_factors = [3**i for i in range(1, int(np.log(max(self.grid_size_tot_unique)) / np.log(3)))]
                        print("Using computed overview factors", overview_factors)
                    dataset.build_overviews(overview_factors, getattr(rasterio.enums.Resampling, overview_type))
                    dataset.update_tags(ns='rio_overview', resampling=overview_type)

        return

    def save_geotiff(self, filename, key, dtype, crs=None, max_files=2, rescale=None, overview_type=None, overview_factors=None):
        '''
        filename: str
            Name of the new file
        key: str
            key of the dataset that should be saved
        dtype: str, np.dtype
            datatype for the resulting data
        crs: str, optional
            The coordinate reference system. Default is rasterio.open(self.elev_source_files[0], 'r').crs
        max_files: int, optional
            Default is 2. Maximum number of files we can automatically break the data into, depending on mismatches of deltas. If
            the number is exceeded an error is raised
        rescale: tuple, optional
            Default is None. data_rescaled = (data - rescale[0]) / (rescale[1] - rescale[0]) * rescale[2]
            This is useful for quantizing the data into an integer format to get a higher compression ratio
        overview_type: str, optional
            Default is None. If not None, will build overviews of the specified type for the file.
        overview_factors: tuple, optional
            Default is (3**1, 3**2, 3**3, ...). Factors for the overview resolutions.
        '''
        zf  = self.out_file_noverlap[key]

        if crs is None:
            crs = rasterio.open(self.elev_source_files[0], 'r').crs

        # Create the geotransform(s) for the large file
        decimals_to_round = 14
        lat_check_ids = self.grid_id2i.max(axis=1)
        dlats = np.unique(np.round(self.index[lat_check_ids, self._i('dlat')], decimals = decimals_to_round))
        lon_check_ids = self.grid_id2i.max(axis=0)
        dlons = np.unique(np.round(self.index[lon_check_ids, self._i('dlon')], decimals = decimals_to_round))
        if (dlats.size > 1) or (dlons.size > 1):
            raise NotImplementedError

        top = self.index[:, self._i('top')].max()
        bottom = self.index[:, self._i('bottom')].min()
        left = self.index[:, self._i('left')].min()
        right = self.index[:, self._i('right')].max()
        dlat = (bottom - top) / self.grid_size_tot_unique[0]
        dlon = (right - left) / self.grid_size_tot_unique[1]
        transform = rasterio.transform.Affine.translation(left, top) * rasterio.transform.Affine.scale(dlon, dlat)
        with rasterio.open(
                    filename, 'w', compress="LZW",
                    width=self.grid_size_tot_unique[1], height=self.grid_size_tot_unique[0],
                    tiled=True, blockxsize=512, blockysize=512,
                    count=1,
                    dtype=dtype,
                    driver='GTiff',
                    bigtiff=True,
                    crs=crs,
                    transform=transform
                ) as dataset:
            for i in range(int(np.ceil(self.grid_size_tot_unique[0] / 512))):
                print("Writing block %d of %d" % (i, self.grid_size_tot_unique[0] // 512))
                for j in range(int(np.ceil(self.grid_size_tot_unique[1] / 512))):
                    slc = (slice(i * 512, (i + 1) * 512), slice(j * 512, (j + 1) * 512))
                    data = zf[slc]
                    slc = (slice(i * 512, i * 512 + data.shape[0]), slice(j * 512, j * 512 + data.shape[1]))
                    window = rasterio.windows.Window.from_slices(*slc)
                    if rescale:
                        data = (data - rescale[0]) / (rescale[1] - rescale[0]) * rescale[2]
                    dataset.write(data.astype(dtype), window=window, indexes=1)
            dataset.update_tags(1, rescale=','.join([str(r) for r in rescale]))

            if overview_type is not None:
                if overview_factors is None:
                    overview_factors = [3**i for i in range(1, int(np.log(max(self.grid_size_tot_unique)) / np.log(3)))]
                dataset.build_overviews(overview_factors, getattr(rasterio.enums.Resampling, overview_type))
                dataset.update_tags(ns='rio_overview', resampling=overview_type)

    def process_overviews(self, out_path, keys=['elev', 'uca', 'aspect', 'slope', 'twi'], overviews=[3, 3**2, 3**3, 3**4, 3**5, 3**6, 3**7]):
        for key in keys:
            last_ov = None
            for ov in overviews:
                print ("Processing overview {}:{}".format(key, ov))
                success = self._calc_overview(out_path, key, ov, last_ov)
                if success == 0:
                    break
                last_ov = ov

    def _calc_overview(self, out_path, key, ov, last_ov):

        new_key = '{}_{}'.format(key.split('_')[0], ov)
        if last_ov is None:
            last_key = key
            last_ov = 1
        else:
            last_key = '{}_{}'.format(key.split('_')[0], last_ov)

        out_file = os.path.join(out_path, last_key)
        new_file = os.path.join(out_path, new_key)
        factor = ov // last_ov

        zf =  zarr.open(out_file)
        shape = zf.shape
        chunks = zf.chunks
        if np.any([c == 0 for c in chunks]):
            chunks = shape.copy()
            # return 0

        n_files = [np.ceil(s / c) for s, c in zip(shape, chunks)]

        new_shape = [int(np.ceil(s / (factor))) for s in shape]
        new_chunks = [int(max((factor), np.ceil(ns / n))) for ns, n in zip(new_shape, n_files)]
        if np.any([n <= factor for n in new_shape]):
            return 0

        # Making a new file with the required shape/size
        print("Making a new file with shape {} --> {} and chunks {} --> {}".format(shape, new_shape, chunks, new_chunks))
        zf_new = zarr.open(new_file, shape=new_shape, chunks=new_chunks, mode='a', dtype=self.dtype)

        # Creating list of inputs, populate kwds
        kwds = []
        row = -1
        while (row * new_chunks[0]) < new_shape[0]:
            row += 1
            col = -1
            while (col * new_chunks[1]) < new_shape[1]:
                col += 1
                kwd = dict(fn=out_file, out_fn=new_file, factor=factor,
                           in_slice=(slice(row * new_chunks[0] * factor, (row + 1) * new_chunks[0] * factor),
                                     slice(col * new_chunks[1] * factor, (col + 1) * new_chunks[1] * factor)),
                         out_slice=(slice(row * new_chunks[0], (row + 1) * new_chunks[0]),
                                     slice(col * new_chunks[1], (col + 1) * new_chunks[1]))
                         )
                kwds.append(kwd)
        success = self.queue_processes(calc_overview, kwds)
        # self.out_file['success'][:, 3] = success
        return success

    def process_elevation(self, indices=None):
        # TODO: Also find max, mean, and min elevation on edges
        # initialize zarr output
        out_file = os.path.join(self.out_path, 'elev')
        zf = zarr.open(out_file, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a', dtype=self.dtype)
        out_file_success = os.path.join(self.out_path, 'success')
        success = zarr.open(out_file_success, shape=(self.n_inputs, 4), mode='a', dtype=bool)

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn=out_file, out_slice=self.grid_slice[i], dp_kwargs=self.dem_proc_kwargs)
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_elev_cond, kwds, success[:, 0])
        self.out_file['success'][:, 0] = success
        return success

    def process_aspect_slope(self):
        # initialize zarr output
        out_aspect = os.path.join(self.out_path, 'aspect')
        zf = zarr.open(out_aspect, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a', dtype=self.dtype)
        out_slope = os.path.join(self.out_path, 'slope')
        zf1 = zarr.open(out_slope, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a', dtype=self.dtype)

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_aspect=out_aspect,
                     out_fn_slope=out_slope,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i],
                     dtype=self.dtype,
                     dp_kwargs=self.dem_proc_kwargs)
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_aspect_slope, kwds, success=self.out_file['success'][:, 1],
                intermediate_fun=self.compute_grid_overlaps)
        self.out_file['success'][:, 1] = success
        return success

    def process_uca(self):
        # initialize zarr output
        out_uca = os.path.join(self.out_path, 'uca')
        zf = zarr.open(out_uca, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a',
                dtype=self.dtype)
        out_uca_edges = os.path.join(self.out_path, 'uca_edges')
        zf = zarr.open(out_uca_edges, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a',
                dtype=self.dtype, fill_value=0)
        out_edge_done = os.path.join(self.out_path, 'edge_done')
        zf1 = zarr.open(out_edge_done, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a', dtype=bool)
        out_edge_todo = os.path.join(self.out_path, 'edge_todo')
        zf2 = zarr.open(out_edge_todo, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a', dtype=bool)

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_uca=out_uca,
                     out_fn_todo=out_edge_todo,
                     out_fn_done=out_edge_done,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i],
                     edge_slice=self.edge_data[i],
                     dtype=self.dtype,
                     dp_kwargs=self.dem_proc_kwargs)
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_uca, kwds, success=self.out_file['success'][:, 2])
        self.out_file['success'][:, 2] = success
        return success

    def update_uca_edge_metrics(self, out_uca=None, index=None):
        if index is None:
            index = range(self.n_inputs)
        if out_uca is None:
            out_uca = os.path.join(self.out_path, 'uca')
        else:
            out_uca = os.path.join(self.out_path, out_uca)
        out_edge_done = os.path.join(self.out_path, 'edge_done')
        out_edge_todo = os.path.join(self.out_path, 'edge_todo')
        out_metrics = os.path.join(self.out_path, 'uca_edge_metrics')
        zf = zarr.open(out_metrics, shape=(self.n_inputs, 2),
                mode='a', dtype=self.dtype,
                fill_value=0)

        kwds = [dict(
                     fn_uca=out_uca,
                     fn_done=out_edge_done,
                     fn_todo=out_edge_todo,
                     slc=self.grid_slice[i],
                     edge_slc=self.edge_data[i])
                for i in index]

        metrics = self.queue_processes(calc_uca_ec_metrics, kwds)
        for i, ind in enumerate(index):
            if metrics[i][0] is None:
                continue
            zf[ind] = metrics[i]
        return zf[:]

    def process_uca_edges(self, mets_type=0):
        out_uca = os.path.join(self.out_path, 'uca')
        out_uca_edges = os.path.join(self.out_path, 'uca_edges')
        out_edge_done = os.path.join(self.out_path, 'edge_done')
        out_edge_todo = os.path.join(self.out_path, 'edge_todo')

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_uca=out_uca,
                     out_fn_uca_edges=out_uca_edges,
                     out_fn_todo=out_edge_todo,
                     out_fn_done=out_edge_done,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i],
                     edge_slice=self.edge_data[i],
                     dtype=self.dtype,
                     dp_kwargs=self.dem_proc_kwargs)
                for i in range(self.n_inputs)]
        # update metrics and decide who goes first
        mets = self.update_uca_edge_metrics('uca')
        if mets.shape[0] == 1:
            I = np.zeros(1, int)
        else:
            I = np.argpartition(-mets[:, mets_type], self.n_workers * 2)

        # Helper function for updating metrics
        def check_mets(finished):
            check_met_inds = []
            for f in finished:
                i, j, k = self.grid_id[f]
                check_met_inds.append(f)
                # left
                if j > 0: check_met_inds.append(self.grid_id2i[i, j-1])
                # right
                if j < self.grid_id2i.shape[1] - 1: check_met_inds.append(self.grid_id2i[i, j+1])
                # top
                if i > 0: check_met_inds.append(self.grid_id2i[i-1, j])
                # bottom
                if i < self.grid_id2i.shape[0] - 1: check_met_inds.append(self.grid_id2i[i+1, j])
            check_met_inds = np.unique(check_met_inds)
            check_met_inds = check_met_inds[check_met_inds != -1].tolist()
            mets = self.update_uca_edge_metrics('uca', check_met_inds)
            if mets.shape[0] == 1:
                I = np.zeros(1, int)
            else:
                I = np.argpartition(-mets[:, mets_type], self.n_workers * 2)
            return mets, I

        I_old = np.zeros_like(I)
        # Non-parallel case (useful for debuggin)
        if self.n_workers == 1:
            count = 0
            while np.any(I_old != I):
                if self._debug:# and count >= 88:
                    from matplotlib.pyplot import figure, subplot, pcolormesh, title, pause, axis, colorbar, clim, show, imshow
                    figure(figsize=(8, 4), dpi=200)
                    subplot(221)
                    imshow(self.out_file['uca'][:] + self.out_file['uca_edges'])
                    title('uca({}) [{}] --> [{}]'.format(count, I_old[0], I[0]))
                    axis('scaled')
                    subplot(222)
                    imshow(self.out_file['edge_todo'][:] *2.0 + self.out_file['edge_done'][:] *1.0, cmap='jet')
                    title('edge_done (1) edge_todo (2)')
                    axis('scaled')
                    clim(0, 3)
                    colorbar()
                    subplot(223)
                    imshow(self.out_file['aspect'][:] * 180 / np.pi, cmap='twilight')
                    title('slope')
                    axis('scaled')
                    clim(0, 360)
                    colorbar()
                    subplot(224)
                    if self._true_uca is None:
                        imshow(self.out_file['aspect'][:]*180/np.pi, cmap='twilight')
                        title('aspect')
                        clim(0, 360)
                    else:
                        imshow(self.out_file['uca'][:, :] + self.out_file['uca_edges'][:, :] - self._true_uca_overlap)
                        title('My - True')
                        clim(None, None)
                    axis('scaled')
                    colorbar()

                    if 0:#self._true_uca is not None:
                        fig = figure(figsize=(8, 4), dpi=200).number
                        subplot(221)
                        imshow(self.out_file['uca'][:, :] + self.out_file['uca_edges'][:, :] - self._true_uca_overlap)
                        title('My - True')
                        colorbar()
                        clim(None, None)
                        axis('scaled')
                        subplot(222)
                        imshow(self.out_file['uca_edges'][:, :])
                        title('Edges')
                        clim(None, None)
                        colorbar()
                        axis('scaled')
                        subplot(223)
                        imshow(self._true_uca_overlap - self.out_file['uca'][:, :])
                        title('True Eges')
                        clim(None, None)
                        colorbar()
                        axis('scaled')
                        subplot(224)
                        imshow(self.out_file['uca'][:, :])
                        title('my')
                        clim(None, None)
                        colorbar()
                        axis('scaled')



                    show()
                count += 1
                print('Count {}'.format(count))
                s = calc_uca_ec(**kwds[I[0]])
                if s[0] == 0:
                    print (s[1])
                I_old[:] = I[:]
                mets, I = check_mets([I[0]])
            return mets

        # create pool and queue
        pool = Pool(processes=self.n_workers)

        # submit workers
        active = I[:self.n_workers * 2].tolist()
        active = [a for a in active if mets[a, 0] > 0]
        res = [pool.apply_async(calc_uca_ec, kwds=kwds[i]) for i in active]
        print ("Starting with {}".format(active))
        while res or np.any(I_old != I):
            # monitor and submit new workers as needed
            finished = []
            finished_res = []
            for i, r in enumerate(res):
                try:
                    s = r.get(timeout=0.001)
                    finished.append(active[i])
                    finished_res.append(i)
                    if s[0] == 0:
                        print (s[1])
                except (mpTimeoutError, TimeoutError) as e:
                    print('.', end='')
                    pass
            if not finished: continue
            mets, I = check_mets(finished)
            I_old[:] = I[:]
            active = [a for a in active if a not in finished]
            res = [r for i, r in enumerate(res) if i not in finished_res]
            candidates = I[:self.n_workers * 2].tolist()
            candidates = [c for c in candidates if c not in active and mets[c, 0] > 0][:(self.n_workers * 2 - len(res))]
            res.extend([pool.apply_async(calc_uca_ec, kwds=kwds[i]) for i in candidates])
            active.extend(candidates)
            print ("Added {}, active {}".format(candidates, active))
            time.sleep(1)
        pool.close()
        pool.join()

        mets = self.update_uca_edge_metrics('uca')
        return mets

    def queue_processes(self, function, kwds, success=None, intermediate_fun=lambda:None):
        # initialize success if not created
        if success is None:
            success = [0] * len(kwds)

        # create pool and queue
        if self.n_workers == 1:
            intermediate_fun()
            for i, kwd in enumerate(kwds):
                if not success[i]:
                    s = function(**kwd)
                    success[i] = s[0]
                    print (s)
                    print (kwd)
            return success

        pool = Pool(processes=self.n_workers)

        # submit workers
        print ("Sumitting workers", end='...')
        #success = [function(**kwds[i])
        #        for i in range(self.n_inputs) if indices[i]]

        res = [pool.apply_async(function, kwds=kwd)
                for i, kwd in enumerate(kwds) if not success[i]]
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

        out_twi = os.path.join(self.out_path, 'twi')
        # Initialize the zarr array with the correct sized chunks
        zf = zarr.open(out_twi, shape=self.grid_size_tot, chunks=self.grid_chunk, mode='a',
                dtype=self.dtype)

        # populate kwds
        kwds = [dict(fn=self.elev_source_files[i],
                     out_fn_twi=out_twi,
                     out_fn=self.out_path,
                     out_slice=self.grid_slice[i],
                     dtype=self.dtype,
                     dp_kwargs=self.dem_proc_kwargs)
                for i in range(self.n_inputs)]

        success = self.queue_processes(calc_twi, kwds, success=self.out_file['success'][:, 3])
        self.out_file['success'][:, 3] = success
        return success