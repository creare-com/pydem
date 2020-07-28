# -*- coding: utf-8 -*-
"""
   Copyright 2015 Creare

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Digital Elevation Model Processing Module
==========================================

This module implements Tarboton (96,
http://www.neng.usu.edu/cee/faculty/dtarb/96wr03137.pdf) Digital Elevation
Model (DEM) processing algorithms for calculating the magnitude and direction
(or aspect) of slopes given the DEM elevation data. This implementation
takes into account the supplied coordinate system, so it does not make an
assumption of a rectangular grid for improved accuracy.

It also implements a novel Upstream Contributing Area (UCA) calculation
algorithm that can be used on multiple input files, and accurately handle
the fluxes at edges of files.

Finally, it can calculate the Topographic Wetness Index (TWI) based on the UCA
and slope magnitude. Flats and no-data areas are in-painted, the maximum
value for TWI is capped, and the output is multiplied by 10 to get higher
accuracy when storing the array as an integer.

Usage Notes
-------------
This module consists of 3 classes, and 3 helper functions. General users should
only need to be concerned with the DEMProcessor class.

It has only been tested in USGS geotiff files.

It presently only supports WGS84 coordinate systems

Development Notes
------------------

Created on Wed Jun 18 14:19:04 2014

@author: mpu
"""

import warnings
import gc
import numpy as np


import os
import subprocess
import traitlets as tl
import traittypes as tt
import scipy.sparse as sps
import scipy.ndimage as spndi

from .taudem import taudem
from .test_pydem import get_test_data, make_file_names
from .utils import (mk_dx_dy_from_geotif_layer, get_fn,
                   make_slice, is_edge, grow_obj, find_centroid, get_distance,
                   get_border_index, get_border_mask, get_adjacent_index,
                   dem_processor_from_raster_kwargs)

try:
    from .cyfuncs import cyutils
    CYTHON = True
except:
    CYTHON = False
    warnings.warn("Cython functions are not compiled. UCA cannot be calculated."
                  " Consider compiling cython functions using: "
                  "python setup.py build_ext --inplace", RuntimeWarning)
# CYTHON = False

# A test aspect ration between dx and dy coordinates
TEST_DIV = 1/1.1  # 0.001
FLAT_ID = np.nan  # This is the fill value for flats in float arrays
FLAT_ID_INT = -1  # This is the fill value for flats in int arrays

# Used to extend flat region downstream when calculating angle
FLATS_KERNEL1 = np.ones((3, 3), bool)
FLATS_KERNEL2 = np.ones((3, 3), bool)  # Convolution used to find edges to the flats
# This is the only choice for FLATS_KERNEL3 because otherwise it won't play
# nicely with the pit-filling algorithm
FLATS_KERNEL3 = np.ones((3, 3), bool)  # Kernel used to connect flats and edges
FILL_VALUE = -9999  # This is the integer fill value for no-data values


class DEMProcessor(tl.HasTraits):
    """
    This class processes elevation data, and returns the magnitude of slopes,
    the direction of slopes, the upstream contributing area, and the
    topographic wetness index.

    """
    fill_flats = tl.Bool(True)
    fill_flats_below_sea = tl.Bool(False)
    fill_flats_source_tol = tl.Int(1)
    fill_flats_peaks = tl.Bool(True)
    fill_flats_pits = tl.Bool(True)
    fill_flats_max_iter = tl.Int(10)

    drain_pits = tl.Bool(True)
    drain_flats = tl.Bool(False) # will be ignored if drain_pits is True
    drain_pits_max_iter = tl.Int(100)
    drain_pits_max_dist = tl.Int(20) # coordinate space
    drain_pits_max_dist_XY = tl.Float(None, allow_none=True) # real space

    # When resolving drainage across edges, if maximum UCA is reached, should
    # edge be marked as completed?
    apply_uca_limit_edges = tl.Bool(False)
    # When calculating TWI, should TWI be limited to max value?
    apply_twi_limits = tl.Bool(False)
    # When calculating TWI, should UCA be limited to max value?
    apply_twi_limits_on_uca = tl.Bool(False)

    direction = tt.Array(None, allow_none=True)  # Direction of slope in radians
    mag = tt.Array(None, allow_none=True)  # magnitude of slopes m/m
    uca = tt.Array(None, allow_none=True)  # upstream contributing area
    twi = tt.Array(None, allow_none=True)  # topographic wetness index
    elev = tt.Array(None, allow_none=True)  # elevation data
    A = None  # connectivity matrix

    # Gives the quadrant used for determining the d_infty mag/direction
    section = None  # save for debugging purposes, not useful output otherwise
    # Give the proportion of the area to drain to the first pixel in quadrant
    proportion = None  # Also saved for debugging
    done = tl.Instance(np.ndarray, None)  # Marks if edges are done

    plotflag = tl.Bool(False)  # Debug plots

    dX = tt.Array(None, allow_none=True)  # delta x
    dY = tt.Array(None, allow_none=True)  # delta y
    
    bounds = tl.List()
    transform = tl.List()

    def __init__(self, elev_fn=None, **kwargs):
        if elev_fn:
            kwds = (dem_processor_from_raster_kwargs(elev_fn))
            kwds.update(kwargs)
            kwargs = kwds
        
        super().__init__(**kwargs)

    @tl.default('dX')
    def _default_dX(self):
        return np.ones(self.elev.shape[0] - 1) / self.elev.shape[1]  # dX only changes in latitude

    @tl.default('dY')
    def _default_dY(self):
        return np.ones(self.elev.shape[0] - 1) / self.elev.shape[0]

    flats = tl.Instance(np.ndarray, None)  # Boolean array indicating location of flats

    uca_saturation_limit = tl.Float(32)  # units of area
    twi_min_slope = tl.Float(1e-3)  # Used for TWI max limiting
    twi_min_area = tl.Float(np.inf)  # Finds min area in tile
    # Mostly deprecated, but maximum number of iterations used to try and
    # resolve circular drainage patterns (which should never occur)
    circular_ref_maxcount = tl.Int(50)
    # Default maximum surface area of pit artifacts resulting from quantization
    # that will be filled
    maximum_pit_area = tl.Float(32)

    # The pixel coordinates for the different facets used to calculate the
    # D_infty magnitude and direction (from Tarboton) uses ij, not xy coordinates
    #  -1,-1 -1,0  -1,1
    #        \ |  /
    #         \|/
    # 0,-1 -- 0,0 -- 0,1
    #         /|\
    #       /  |  \
    #  1,-1   1,0   1,1
    #
    #       [2] [1] 
    #        \ |  /
    #    [3]  \|/  [0]
    #      --     -- 
    #    [4]  /|\  [7]
    #       /  |  \
    #       [5] [6]
    facets = tl.List([
             [(0, 0), (0, 1), (-1, 1)],
             [(0, 0), (-1, 0), (-1, 1)],
             [(0, 0), (-1, 0), (-1, -1)],
             [(0, 0), (0, -1), (-1, -1)],
             [(0, 0), (0, -1), (1, -1)],
             [(0, 0), (1, 0), (1, -1)],
             [(0, 0), (1, 0), (1, 1)],
             [(0, 0), (0, 1), (1, 1)],
    ])
    # Helper for magnitude/direction calculation (modified from Tarboton)
    ang_adj = tt.Array([
                       [0, 1],
                       [1, -1],
                       [1, 1],
                       [2, -1],
                       [2, 1],
                       [3, -1],
                       [3, 1],
                       [4, -1]
                       ])

    def get_fn(self, name=None):
        return get_fn(self.elev, name)

    def get_full_fn(self, name, rootpath='.'):
        return os.path.join(rootpath, name, self.get_fn(name))

    def load_array(self, fn, name):
        """
        Can only load files that were saved in the 'raw' format.
        Loads previously computed field 'name' from file
        Valid names are 'mag', 'direction', 'uca', 'twi'
        """

        if os.path.exists(fn + '.npz'):
            array = np.load(fn + '.npz')
            try:
                setattr(self, name, array['arr_0'])
            except Exception as e:
                print(e)
            finally:
                array.close()

        else:
            raise RuntimeError("File %s does not exist." % (fn + '.npz'))

    def load_elevation(self, fn):
        """Loads pre-computed (interpolated/filled) elevation from file
        """
        self.load_array(fn, 'elev')

    def load_slope(self, fn):
        """Loads pre-computed slope magnitude from file
        """
        self.load_array(fn, 'mag')

    def load_direction(self, fn):
        """Loads pre-computed slope direction from file
        """
        self.load_array(fn, 'direction')

    def load_uca(self, fn):
        """Loads pre-computed uca from file
        """
        self.load_array(fn, 'uca')

    def find_flats(self):
        self.flats = self.mag == FLAT_ID_INT

    def _fill_flat(self, roi, out, region, edge, it=0, debug=False):
        e = roi[region][0]

        # 1-pixel special cases (3x3, 3x2, 2x3, and 2x2)
        if roi.size <= 9 and region.sum() == 1:
            source = roi > e
            n = source.sum()
            if n == roi.size-1:
                # pit
                pass
            elif n > 0:
                # special fill case
                out[region] += min(1.0, (roi[source].min() - e)) - 0.01
            elif self.fill_flats_peaks:
                # small peak
                out[region] += 0.5

            return

        # get source and drain masks
        border = get_border_mask(region)
        drain = border & (roi == e)
        source = border & (roi > e)

        replace = None

        # update source and set eH (high elevation for interpolation)

        # if (it == 0 and drain.any() and (region & edge).any() and (region & edge & get_border_mask(source) & ~get_border_mask(drain)).any()):
        #     # Downstream side of river beds that cross an edge: drain from edge
        #     replace = region & edge & get_border_mask(source) & ~get_border_mask(drain)
        #     eH = min(e + 0.5, (e+roi[source].min())/2.0)
        #     out[replace] = eH
        #     source = replace
        # elif source.any():
        if source.any():
            # Normal case: interpolate from shallow sources (non-cliffs)
            e_source = roi[source].min()
            eH = min(e + 1.0, e_source)
            source &= (roi <= e_source + self.fill_flats_source_tol)
        elif self.fill_flats_peaks:
            # Mountain peaks: drain from a center point in the peak
            eH = e + 0.5
            centroid = find_centroid(region)
            out[centroid] = eH
            source[centroid] = True
            replace = source
        else:
            return

        # update drain
        if drain.any():
            # Normal case
            pass
        elif (region & edge).any():
            # Upstream side of river beds that cross an edge: drain to edge
            replace = drain = region & edge
            if not (region & ~drain).any():
                return
        elif self.fill_flats_pits:
            # Pit area
            centroid = find_centroid(region)
            drain[centroid] = True
            replace = drain
        else:
            return

        # interpolate flat area
        dH = get_distance(region, source)
        dL = get_distance(region, drain)
        if replace is None: interp = region
        else: interp = region & ~replace
        out[interp] = (eH*dL[interp]**2 + e*dH[interp]**2) / (dL[interp]**2 + dH[interp]**2)

        if it >= self.fill_flats_max_iter:
            warnings.warn("Maximum iterations exceeded for fill_flats")
            return

        # iterate to fill remaining flat areas (created during interpolation)
        flat = (spndi.minimum_filter(out, (3, 3)) >= out) & region
        if flat.any():
            out2 = out.copy()
            flats, n = spndi.label(flat, structure=FLATS_KERNEL3)
            for i, _obj in enumerate(spndi.find_objects(flats)):
                obj = grow_obj(_obj, roi.shape)
                self._fill_flat(out[obj], out2[obj], flats[obj]==i+1, edge[obj], it=it+1)
            out = out2

    def calc_fill_pit_artifacts(self):
        """
        Correct pits, up to a maximum surface area, that are artifacts of elevation
        quantization.
        """

        self.filledPits = self.elev.copy()
        if self.fill_flats_below_sea:
            sea_mask = self.elev != 0
        else:
            sea_mask = self.elev > 0
        flat = (spndi.minimum_filter(self.elev, (3, 3)) >= self.elev) & sea_mask
        flats, n = spndi.label(flat, structure=FLATS_KERNEL3)
        objs = spndi.find_objects(flats)
        cnt = 0
        max_size = {}
        for i, _obj in enumerate(objs):
            obj = grow_obj(_obj, self.elev.shape)
            if not (self.elev[obj].shape[0] == self.elev[_obj].shape[0] + 2 and self.elev[obj].shape[1] == self.elev[_obj].shape[1] + 2):
                continue
            mask = flats[obj] == i+1
            edge_mask = spndi.maximum_filter(mask, (3, 3)) ^ mask
            my_elev = self.elev[obj][mask][0]
            edge_condition = np.all(self.elev[obj][edge_mask] - 1 == my_elev)
            if edge_condition and mask.sum() <= self.maximum_pit_area:
                if mask.sum() not in max_size:
                    max_size[mask.sum()] = 0
                max_size[mask.sum()] += 1
                cnt = cnt + 1
                self.filledPits[obj] += 1*mask
        self.elev = self.filledPits


    def calc_fill_flats(self):
        """
        Fill/interpolate flats, if they have not yet been filled. This
        is to be done prior to self.calc_slopes_directions.
        """

        if self.maximum_pit_area:
            self.calc_fill_pit_artifacts()

        # fill/interpolate flats first
        data = np.ma.filled(self.elev.astype('float64'), np.nan)
        filled = data.copy()
        edge = np.ones_like(data, bool)
        edge[1:-1, 1:-1] = False
        if self.fill_flats_below_sea: sea_mask = data != 0
        else: sea_mask = data > 0
        flat = (spndi.minimum_filter(data, (3, 3)) >= data) & sea_mask
        # Exclude corners
        flat[0, 0] = False
        flat[-1, 0] = False
        flat[0, -1] = False
        flat[-1, -1] = False
        # TODO minimum filter behavior with nans?
        flats, n = spndi.label(flat, structure=FLATS_KERNEL3)
        objs = spndi.find_objects(flats)
        for i, _obj in enumerate(objs):
            obj = grow_obj(_obj, data.shape)
            self._fill_flat(data[obj], filled[obj], flats[obj]==i+1, edge[obj])
        self.elev = filled


        # if debug:
        #     from matplotlib import pyplot
        #     from utils import plot_flat
        #     plot_flat(roi, out, region, source, drain, dL, dH)
        #     pyplot.show()

    def calc_slopes_directions(self, plotflag=False):
        """
        Calculates the magnitude and direction of slopes and fills
        self.mag, self.direction
        """

        if self.fill_flats:
            self.calc_fill_flats()

        # %% Calculate the slopes and directions based on the 8 sections from
        # Tarboton http://www.neng.usu.edu/cee/faculty/dtarb/96wr03137.pdf
        print("starting slope/direction calculation")
        self.mag, self.direction = self._slopes_directions(
                self.elev, self.dX, self.dY, 'tarboton')
        # Find the flat regions. This is mostly simple (look for mag < 0),
        # but the downstream pixel at the edge of a flat will have a
        # calcuable angle which will not be accurate. We have to also find
        # these edges and set their magnitude to -1 (that is, the flat_id)
        flats = self._find_flats_edges(self.elev, self.mag, self.direction)
        self.direction[flats] = FLAT_ID_INT
        self.mag[flats] = FLAT_ID_INT
        self.flats = flats

        if plotflag:
            self._plot_debug_slopes_directions()

        gc.collect()  # Just in case
        return self.mag, self.direction

    def _slopes_directions(self, data, dX, dY, method='tarboton'):
        """ Wrapper to pick between various algorithms
        """
        # %%
        if method == 'tarboton':
            return self._tarboton_slopes_directions(data, dX, dY)
        elif method == 'central':
            return self._central_slopes_directions(data, dX, dY)

    def _tarboton_slopes_directions(self, data, dX, dY):
        """
        Calculate the slopes and directions based on the 8 sections from
        Tarboton http://www.neng.usu.edu/cee/faculty/dtarb/96wr03137.pdf
        """

        return _tarboton_slopes_directions(data, dX, dY,
                                           self.facets, self.ang_adj)

    def _central_slopes_directions(self, data, dX, dY):
        """
        Calculates magnitude/direction of slopes using central difference
        """
        shp = np.array(data.shape) - 1

        direction = np.full(data.shape, FLAT_ID_INT, 'float64')
        mag = np.full(direction, FLAT_ID_INT, 'float64')

        ind = 0
        d1, d2, theta = _get_d1_d2(dX, dY, ind, [0, 1], [1, 1], shp)
        s2 = (data[0:-2, 1:-1] - data[2:, 1:-1]) / d2
        s1 = -(data[1:-1, 0:-2] - data[1:-1, 2:]) / d1
        direction[1:-1, 1:-1] = np.arctan2(s2, s1) + np.pi
        mag = np.sqrt(s1**2 + s2**2)

        return mag, direction

    def _find_flats_edges(self, data, mag, direction):
        """
        Extend flats 1 square downstream
        Flats on the downstream side of the flat might find a valid angle,
        but that doesn't mean that it's a correct angle. We have to find
        these and then set them equal to a flat
        """

        i12 = np.arange(data.size).reshape(data.shape)

        flat = mag == FLAT_ID_INT
        flats, n = spndi.label(flat, structure=FLATS_KERNEL3)
        objs = spndi.find_objects(flats)

        f = flat.ravel()
        d = data.ravel()
        for i, _obj in enumerate(objs):
            region = flats[_obj] == i+1
            I = i12[_obj][region]
            J = get_adjacent_index(I, data.shape, data.size)
            f[J] = d[J] == d[I[0]]

        flat = f.reshape(data.shape)
        return flat

    def calc_uca(self, plotflag=False, edge_init_data=None, uca_init=None):
        """Calculates the upstream contributing area.

        Parameters
        ----------
        plotflag : bool, optional
            Default False. If true will plot debugging plots. For large files,
            this will be very slow
        edge_init_data : list, optional
            edge_init_data = [uca_data, done_data, todo_data]
            uca_data : dict
                Dictionary with 'left', 'right', 'top', 'bottom' keys that
                gives the arrays filled with uca data on the edge corresponding
                to the key
            done_data : dict
                As uca_data, but bool array indicating if neighboring tiles
                have computed a finished value for that edge pixel
            todo_data : dict
                As uca_data, but bool array indicating if edges on tile still
                have to be computed
        uca_init : array, optional
            Array with pre-computed upstream contributing area
            (without edge contributions)

        Notes
        -------
        if edge_init_data is given, then the initialized area will be modified
        such that the edges are equal to the edge_init_data.

        If uca_init is given, then the interior of the upstream area will not
        be calculated. Only the information from the edges will be updated.
        Unless the tile is too large so that the calculation is chunked. In
        that case, the whole tile is re-computed.
        """
        if not CYTHON :
            raise RuntimeError("Cython functions need to be compiled to perform UCA computations")
        if self.direction is None:
            self.calc_slopes_directions()

        # Initialize the upstream area
        uca_edge_init = np.zeros(self.elev.shape, 'float64')
        uca_edge_done = np.zeros(self.elev.shape, bool)
        uca_edge_todo = np.zeros(self.elev.shape, bool)
        edge_data, edge_init_todo, edge_init_done = [None] * 3
        if edge_init_data is not None:
            edge_data, edge_init_done, edge_init_todo = edge_init_data
            slices = {'left': (slice(None), slice(0, 1)),
                      'right': (slice(None), slice(-1, None)),
                      'top': (slice(0, 1), slice(None)),
                      'bottom': (slice(-1, None), slice(None))}
            for key, val in slices.items():
                # To initialize and edge it needs to have data and be finished
                uca_edge_done[val] = uca_edge_done[val] \
                    | edge_init_done[key].reshape(uca_edge_init[val].shape)
                uca_edge_init[val] += \
                    (edge_data[key] * edge_init_done[key]).reshape(uca_edge_init[val].shape)
                uca_edge_todo[val] = uca_edge_todo[val]\
                    | edge_init_todo[key].reshape(uca_edge_init[val].shape)
            for key, val in slices.items():
                uca_edge_init[val][~uca_edge_done[val]] = 0

        if uca_init is None:
            self.uca = np.full(self.elev.shape, FLAT_ID_INT, 'float64')
        else:
            self.uca = uca_init.astype('float64')

        if uca_init is None:
            print("Starting uca calculation")
            res = self._calc_uca_chunk(self.elev, self.dX, self.dY,
                                       self.direction, self.mag,
                                       self.flats,
                                       area_edges=uca_edge_init,
                                       plotflag=plotflag)
            self.edge_todo = res[1]
            self.edge_done = res[2]
            self.uca = res[0]
            # Fix the very last pixel on the edges
            self.uca = self.fix_edge_pixels(edge_data, edge_init_done, edge_init_todo, self.uca)            
        else:
            print("Starting edge resolution round: ", end='')
            # last return value will be None: edge_
            area, e2doi, edone, _ = \
                self._calc_uca_chunk_update(self.elev, self.dX, self.dY,
                                            self.direction, self.mag,
                                            self.flats,
                                            area_edges=uca_edge_init,
                                            edge_init=uca_edge_init,
                                            edge_todo=uca_edge_todo,
                                            edge_done=uca_edge_done,
                                            edge_init_data=edge_init_data)
            self.uca += area
            self.edge_todo = e2doi
            self.edge_done = edone
            self.uca = self.fix_self_edge_pixels(edge_data, edge_init_done, edge_init_todo, self.uca)
            self.uca = self.fix_edge_pixels(edge_data, edge_init_done, edge_init_todo, self.uca, False)            

        print('..Done')


        gc.collect()  # Just in case
        return self.uca

    def fix_edge_pixels(self, edge_init_data, edge_init_done, edge_init_todo, uca=None, initialize=True):
        """
        This function fixes the pixels on the very edge of the tile.
        Drainage is calculated if the edge is downstream from the interior.
        If there is data available on the edge (from edge_init_data, for eg)
        then this data is used.

        This is a bit of hack to take care of the edge-values. It could
        possibly be handled through the main algorithm, but at least here
        the treatment is explicit.
        """
        data, dX, dY, direction, flats = \
            self.elev, self.dX, self.dY, self.direction, self.flats
        if uca is None:
            uca = self.uca
        sides = ['left', 'right', 'top', 'bottom']
        # Nodes that will drain their contributions
        slices_o = ((slice(None), slice(1, 2)), (slice(None), slice(-2, -1)),
                    (slice(1, 2), slice(None)), (slice(-2, -1), slice(None)))
        # Edges that will receive the contributions
        slices_d = ((slice(None), slice(0, 1)), (slice(None), slice(-1, None)),
                    (slice(0, 1), slice(None)), (slice(-1, None), slice(None)))

        # The first set of edges will have contributions from two nodes whereas
        # the second set of edges will only have contributinos from one node
        # Each interior node drains to an edge node in one of these configurations
        #              ^ ^  ^^
        #             \ \|  |/ 
        #            <-3 2 1 /
        #            <-4   0 ->   
        #             /5 6 7 -> 
        #             /| |\ \
        #
        indices = {'left': [[3, 4], [2, 5]], 'right': [[0, 7], [1, 6]],
                   'top': [[1, 2], [0, 3]], 'bottom': [[5, 6], [4, 7]]}
        indices_corners = {'top-left': [2, 3], 'top-right': [0, 1],
                           'bottom-left': [4, 5], 'bottom-right': [6, 7]}


        for side, slice_o, slice_d in zip(sides, slices_o, slices_d):
            if not initialize:
                 continue
            # self-initialize values for the edges:
            if side in ['left', 'right']:
                uca[slice_d] = \
                    np.concatenate(([dX[slice_d[0]][0] * dY[slice_d[0]][0]],
                                    dX[slice_d[0]] * dY[slice_d[0]]))\
                    .reshape(uca[slice_d].shape)
            else:
                uca[slice_d] = dX[slice_d[0]][0] * dY[slice_d[0]][0]

        for side, slice_o, slice_d in zip(sides, slices_o, slices_d):
            section, proportion = \
                self._calc_uca_section_proportion(data[slice_o],
                                                  dX[slice_o[0]],
                                                  dY[slice_o[0]],
                                                  direction[slice_o],
                                                  flats[slice_o])
            # Loop over the different types of draining quadrants
            for e in range(2):
                for i in indices[side][e]:
                    ed = self.facets[i][2]
                    ids = np.where(section == i)
                    if ids[0].size == 0:
                         continue
                    ids2ri = [ids[0].copy(), ids[1].copy()] # This structure will store the offset needed for the edge draining
                    ids2r = [ids[0].copy(), ids[1].copy()] # This structure will store the offset needed for the edge draining
                    # fix edges
                    if side in ['left', 'right']:
                        ids2ri[0] += ed[0]
                        # Exclude corners
                        ids2r[0] = np.clip(ids2ri[0], 0, uca.shape[0] - 1)
                        noclip = ids2r[0] == ids2ri[0]
                    else:
                        ids2ri[1] += ed[1]
                        ids2r[1] = np.clip(ids2ri[1], 0, uca.shape[1] - 1)
                        # Exclude corners
                        noclip = ids2r[1] == ids2ri[1]
                        
                    ids2r = tuple(ids2r)

                    if e == 0:
                        uca[slice_d][ids] += uca[slice_o][ids] * proportion[ids]
                        uca[slice_d][ids2r] += uca[slice_o][ids] * (1 - proportion[ids]) * noclip
                    elif e == 1:
                        uca[slice_d][ids2r] += uca[slice_o][ids] * (1 - proportion[ids]) * noclip

        # Do the corners
        for key in indices_corners.keys():
            # Re-initialize corner
            
            slice_d = []
            slice_o = []
            if 'top' in key:
                slice_d.append(slice(0, 1))
                slice_o.append(slice(1, 2))
            else:
                slice_d.append(slice(-1, None))
                slice_o.append(slice(-2, -1))
            if 'left' in key:
                slice_d.append(slice(0, 1))
                slice_o.append(slice(1, 2))
            else:
                slice_d.append(slice(-1, None))
                slice_o.append(slice(-2, -1))
            slice_d = tuple(slice_d)
            slice_o = tuple(slice_o)
            
            uca[slice_d] = dX[slice_d[0]] * dY[slice_d[0]]
            
            section, proportion = \
                self._calc_uca_section_proportion(data[slice_o],
                                                  dX[slice_o[0]],
                                                  dY[slice_o[0]],
                                                  direction[slice_o],
                                                  flats[slice_o])
            if section.item() in indices_corners[key]:
                uca[slice_d] +=  uca[slice_o] * (1 - proportion)
        
        for side, slice_o, slice_d in zip(sides, slices_o, slices_d):
            # Add the edge data from adjacent tiles
            if edge_init_done is not None:
                ids = edge_init_done[side]  # > 0
                if side in ['left', 'right']:
                    uca[slice_d][ids, :] = \
                        edge_init_data[side][ids][:, None]
                else:
                    uca[slice_d][:, ids] = edge_init_data[side][ids]
        return uca
        
    def fix_self_edge_pixels(self, edge_init_data, edge_init_done, edge_init_todo, uca=None):
        """
        This function fixes the pixels on the very edge of the tile by following draining along the edge itself
        It handles edge to edge contributions only. 
        """
        data, dX, dY, direction, flats = \
            self.elev, self.dX, self.dY, self.direction, self.flats
        if uca is None:
            uca = self.uca
        sides = ['left', 'right', 'top', 'bottom']
        # Nodes that will drain their contributions
        slices_o = ((slice(None), slice(1, 2)), (slice(None), slice(-2, -1)),
                    (slice(1, 2), slice(None)), (slice(-2, -1), slice(None)))
        # Edges that will receive the contributions
        slices_d = ((slice(None), slice(0, 1)), (slice(None), slice(-1, None)),
                    (slice(0, 1), slice(None)), (slice(-1, None), slice(None)))

        # The first set of edges will have contributions from two nodes whereas
        # the second set of edges will only have contributinos from one node
        # Each interior node drains to an edge node in one of these configurations
        #              ^ ^  ^^
        #             \ \|  |/ 
        #            <-3 2 1 /
        #            <-4   0 ->   
        #             /5 6 7 -> 
        #             /| |\ \
        #

        indices_self = {'left': [[5, 6], [1, 2]],
                        'top': [[0, 7], [3, 4]]}
        indices_self['right'] = indices_self['left']
        indices_self['bottom'] = indices_self['top']

        for side, slice_o, slice_d in zip(sides, slices_o, slices_d):
            # self-initialize values for the edges:
            if side in ['left', 'right']:
                uca[slice_d] = \
                    np.concatenate(([dX[slice_d[0]][0] * dY[slice_d[0]][0]],
                                    dX[slice_d[0]] * dY[slice_d[0]]))\
                    .reshape(uca[slice_d].shape)
            else:
                uca[slice_d] = dX[slice_d[0]][0] * dY[slice_d[0]][0]
        
        if edge_init_data is not None:
            for side, slice_o, slice_d in zip(sides, slices_o, slices_d):
                # Add the edge data from adjacent tiles
                ids = edge_init_done[side]  # > 0
                if side in ['left', 'right']:
                    uca[slice_d][ids, :] = \
                        edge_init_data[side][ids][:, None]
                else:
                    uca[slice_d][:, ids] = edge_init_data[side][ids]
            
        
        for side, slice_o, slice_d in zip(sides, slices_o, slices_d):
            # Let's propagate data from edge-point to edge-point, but only for edges that drain INTO the interior
            section, proportion = self._calc_uca_section_proportion(
                data[slice_d], dX[slice_d[0]], dY[slice_d[0]], direction[slice_d], flats[slice_d])

            proportion = proportion.ravel()
            # Loop over the different types of draining quadrants
            for e in range(2):
                i = indices_self[side][e]
                ids = np.any([section == ii for ii in i], axis=0).ravel() & (data[slice_d].ravel() > spndi.minimum_filter(data[slice_o].ravel(), 3))
                #if not any(ids):
                    #continue
                if edge_init_data is not None:
                    ids = ids & ~edge_init_done[side]
                ids[-(1 + e)] = False
                rollids = np.roll(ids, 1 - 2 * e)
                rollids[-e] = False
                val = np.zeros(ids.size)
                start_val = None
                for k in range((val.size - 1) * e, val.size * (1 - e) - e, 1 - 2 * e):
                    if ids[k] and not rollids[k]:
                        start_val = uca[slice_d].ravel()[k] * proportion[k]
                        continue
                    if start_val is None:
                        continue
                    val[k] = start_val
                    if rollids[k]:
                        start_val = (start_val + uca[slice_d].ravel()[k]) * proportion[k]
                        if not ids[k]:
                            start_val = None
                
                uca[slice_d] += val.reshape(section.shape)
        return uca
    

    def _calc_uca_chunk_update(self, data, dX, dY, direction, mag, flats,
                               area_edges=None, edge_init=None, edge_todo=None, edge_done=None, edge_init_data=None,
                               plotflag=False):
        """
        Calculates the upstream contributing area due to contributions from
        the edges only.
        """
        # Figure out which section the drainage goes towards, and what
        # proportion goes to the straight-sided (as opposed to diagonal) node.
        section, proportion = self._calc_uca_section_proportion(
            data, dX, dY, direction, flats)

        # Build the drainage or adjacency matrix
        A = self._mk_adjacency_matrix(section, proportion, flats, data, mag, dX, dY)
        B = A
        C = A.tocsr()

        ids = np.zeros(data.shape, bool)
        area = np.zeros(data.shape, 'float64')
        # Set the ids to the edges that are now done, and initialize the
        # edge area
        area[:, 0] = area_edges[:, 0]
        area[:, -1] = area_edges[:, -1]
        area[-1, :] = area_edges[-1, :]
        area[0, :] = area_edges[0, :]

        # Initialize starting ids
        ids = edge_done & edge_todo
        edge_todo = edge_todo & ~edge_done
        edge_todo_tile = None

        ids = ids.ravel()
        ids0 = ids.copy()

        area[flats] = np.nan
        edge_done = ~edge_todo

        edge_todo_i = edge_todo.copy()
        ids_old = np.zeros_like(ids)

        done = np.ones(data.shape, bool)
        done.ravel()[ids] = False
        
        # Drain edges that come from exterior to itself
        area = self.fix_self_edge_pixels(edge_init_data[0], edge_init_data[1], edge_init_data[2], uca=area)

        a = cyutils.drain_connections(
            done.ravel(), ids, B.indptr, B.indices, set_to=False)
        done = a.reshape(done.shape).astype(bool)

        done[np.isnan(data)] = True  # deal with no-data values
        
        # Set all the edges to "done". This ensures that another edges does not stop
        # the propagation of data
        done[:, 0] = True
        done[:, -1] = True
        done[0, :] = True
        done[-1, :] = True
        #
        ids = ids0.copy()
        ids_old[:] = 0

        a, b, c, d = cyutils.drain_area(area.ravel(),
                                        done.ravel(),
                                        ids,
                                        B.indptr, B.indices, B.data,
                                        C.indptr, C.indices,
                                        area.shape[0], area.shape[1],
                                        skip_edge=True)
        area = a.reshape(area.shape)
        done = b.reshape(done.shape)

        # Rather unfortunately, we still have to follow through the boolean
        # edge_todo matrix...
        ids = edge_todo.copy().ravel()
        # %%
        a = cyutils.drain_connections(edge_todo.ravel(),
                                      ids, B.indptr, B.indices,
                                      set_to=True)
        edge_todo = a.reshape(edge_todo.shape).astype(bool)

        area[flats] = np.nan
        edge_done = ~edge_todo
        
        if plotflag:
            self._plot_connectivity(A, (done.astype('float64') is False)
                                    + flats.astype('float64') * 2, [0, 3])        
        
        return area, edge_todo_i, edge_done, edge_todo_tile

    def _calc_uca_chunk(self, data, dX, dY, direction, mag, flats,
                        area_edges, plotflag=False, edge_todo_i_no_mask=True):
        """
        Calculates the upstream contributing area for the interior, and
        includes edge contributions if they are provided through area_edges.
        """
        # %%
        # Figure out which section the drainage goes towards, and what
        # proportion goes to the straight-sided (as opposed to diagonal) node.
        section, proportion = self._calc_uca_section_proportion(
            data, dX, dY, direction, flats)

        # Build the drainage or adjacency matrix
        A = self._mk_adjacency_matrix(section, proportion, flats, data, mag, dX, dY)
        B = A.tocsr()

        colsum = np.array(A.sum(1)).ravel()
        ids = colsum == 0  # If no one drains into me

        area = (dX * dY)
        area = np.concatenate((area[0:1], area)).reshape(area.size+1, 1)
        
        #area = ((dX[1:] + dX[:-1]) / 2 * (dY[1:] + dY[:-1]) / 2)
        #dX0 = (dX[0] - (dX[1] - dX[0]) + dX[0]) / 2
        #dX0 = (3 * dX[0] - dX[1]) / 2
        #dY0 = (3 * dX[0] - dX[1]) / 2
        #area0 = dX0 * dY0
        #areaN = (3 * dX[-1] - dX[-2]) / 2 * (3 * dX[-1] - dX[-2]) / 2
        #area = np.concatenate(([area0], area, [areaN])).reshape(area.size+2, 1)

        # Record minimum area
        min_area = np.nanmin(area)
        self.twi_min_area = min(self.twi_min_area, min_area)
        
        area = area.repeat(data.shape[1], 1)
        # Set the edge areas to zero, will add those contributions later
        area[:, 0] = area_edges[:, 0]
        area[:, -1] = area_edges[:, -1]
        area[-1, :] = area_edges[-1, :]
        area[0, :] = area_edges[0, :]
        # These edges are done, they have been drained already
        ids[area_edges.ravel() > 0] = True

        done = np.zeros(data.shape, bool)
        done.ravel()[ids] = True
        # deal with no-data values
        done[1:-1, 1:-1] = done[1:-1, 1:-1]

        # Check the inlet edges
        edge_todo = np.zeros_like(done)
        ids_ed = np.arange(data.size).reshape(data.shape)
        # left
        edge_todo[:, 0] = (A[:, ids_ed[:, 0]].sum(0) > 0) \
            & (area_edges[:, 0] == 0)
        edge_todo[:, -1] = (A[:, ids_ed[:, -1]].sum(0) > 0) \
            & (area_edges[:, -1] == 0)
        edge_todo[0, :] = (A[:, ids_ed[0, :]].sum(0) > 0) \
            & (area_edges[0, :] == 0)
        edge_todo[-1, :] = (A[:, ids_ed[-1, :]].sum(0) > 0) \
            & (area_edges[-1, :] == 0)

        # Will do the tile-level doneness
        edge_todo_i_no_mask = edge_todo.copy() & edge_todo_i_no_mask
        edge_todo_no_mask = edge_todo_i_no_mask.copy()  # tile-level doneness
        edge_todo[np.isnan(data)] = False  # Don't do masked areas
        # Initialize done edges
        edge_todo_i = edge_todo.copy()
        # %%
        count = 1

        # Pack for Cython function
        area_ = area.ravel()
        done_ = done.ravel()
        edge_todo_ = edge_todo.astype('float64').ravel()
        edge_todo_no_mask_ = edge_todo_no_mask.astype('float64').ravel()
        data_ = data.ravel()

        done_ = done.ravel()
        done_sum = 0
        max_elev = -1
        while (np.any(~done_) and count < self.circular_ref_maxcount)\
                and done_sum != done_.sum():
            print(".", end='') #done_sum, done_.sum(), max_elev, count,
            import sys;sys.stdout.flush()
            done_sum = done_.sum()
            count += 1
            area_, done_, edge_todo_, edge_todo_no_mask_ = cyutils.drain_area(
                area_, done_, ids,
                A.indptr, A.indices, A.data, B.indptr, B.indices,
                area.shape[0], area.shape[1],
                edge_todo_, edge_todo_no_mask_)

            ids[:] = False
            max_elev = (data_ * (~done_)).max()
            ids[((data_ * (~done_) - max_elev) / max_elev > -0.01)] = True

        # Unpack from CYTHON function
        area = area_.reshape(area.shape)
        done = done_.reshape(done.shape)
        edge_todo = edge_todo_.reshape(edge_todo.shape).astype(bool)
        edge_todo_no_mask = edge_todo_no_mask_.reshape(edge_todo_no_mask.shape).astype(bool)

        area[flats] = np.nan

        edge_done = ~edge_todo
        edge_done[np.isnan(data)] = True  # Don't do masked areas

        if self.apply_uca_limit_edges:
            # 2x because of bifurcations (maybe should be more than 2x, but
            # should be ok
            edge_done[area > self.uca_saturation_limit * 2 * min_area] = True

        # %%
        if plotflag:
            # TODO DTYPE
            self._plot_connectivity(A, (done.astype('float64') is False)
                                    + flats.astype('float64') * 2, [0, 3])
        return area, edge_todo_i, edge_done, edge_todo_i_no_mask, edge_todo_no_mask

    def _drain_step(self, A, ids, area, done, edge_todo):
        """
        Does a single step of the upstream contributing area calculation.
        Here the pixels in ids are drained downstream, the areas are updated
        and the next set of pixels to drain are determined for the next round.
        """
        # Only drain to cells that have a contribution
        A_todo = A[:, ids.ravel()]
        colsum = np.array(A_todo.sum(1)).ravel()
        # Only touch cells that actually receive a contribution
        # during this stage
        ids_new = colsum != 0
        # Is it possible that I may drain twice from my own cell?
        # -- No, I don't think so...
        # Is it possible that other cells may drain into me in
        # multiple iterations -- yes
        # Then say I check for when I'm done ensures that I don't drain until
        # everyone has drained into me
        area.ravel()[ids_new] += (A_todo[ids_new, :]
                                  * (area.ravel()[ids].ravel()))
        edge_todo.ravel()[ids_new] += (A_todo[ids_new, :]
                                       * (edge_todo.ravel()[ids].ravel()))
        # Figure out what's left to do.
        done.ravel()[ids] = True

        colsum = A * (~done.ravel())
        ids = colsum == 0
        # Figure out the new-undrained ids
        ids = ids & (~done.ravel())

        return ids, area, done, edge_todo

    def _calc_uca_section_proportion(self, data, dX, dY, direction, flats):
        """
        Given the direction, figure out which nodes the drainage will go
        toward, and what proportion of the drainage goes to which node
        """
        shp = np.array(data.shape) - 1

        facets = self.facets
        adjust = self.ang_adj[:, 1]

        d1, d2, theta = _get_d1_d2(dX, dY, 0, facets[0][1], facets[0][2], shp)
        if dX.size > 1:
            theta = np.row_stack((theta[0, :], theta, theta[-1, :]))
        # Which quadrant am I in?
        section = ((direction / np.pi * 2.0) // 1).astype('int8') # TODO DTYPE
        # Gets me in the quadrant
        quadrant = (direction - np.pi / 2.0 * section)
        proportion = np.full_like(quadrant, np.nan)
        # Now which section within the quadrant
        section = section * 2 \
            + (quadrant > theta.repeat(data.shape[1], 1)) * (section % 2 == 0) \
            + (quadrant > (np.pi/2 - theta.repeat(data.shape[1], 1))) \
            * (section % 2 == 1)  # greater than because of ties resolution b4

        # %%    Calculate proportion
        # As a side note, it's crazy to me how:
        #    _get_d1_d2 needs to use indices 0, 3, 4, 7,
        #    section uses even/odd (i.e. % 2)
        #    proportion uses indices (0, 1, 4, 5) {ALl of them different! ARG!}
        I1 = (section == 0) | (section == 1) | (section == 4) | (section == 5)
    #    I1 = section % 2 == 0
        I = I1 & (quadrant <= theta.repeat(data.shape[1], 1))
        proportion[I] = quadrant[I] / theta.repeat(data.shape[1], 1)[I]
        I = I1 & (quadrant > theta.repeat(data.shape[1], 1))
        proportion[I] = (quadrant[I] - theta.repeat(data.shape[1], 1)[I]) \
            / (np.pi / 2 - theta.repeat(data.shape[1], 1)[I])
        I = (~I1) & (quadrant <= (np.pi / 2 - theta.repeat(data.shape[1], 1)))
        proportion[I] = (quadrant[I]) \
            / (np.pi / 2 - theta.repeat(data.shape[1], 1)[I])
        I = (~I1) & (quadrant > (np.pi / 2 - theta.repeat(data.shape[1], 1)))
        proportion[I] = (quadrant[I] - (np.pi / 2 - theta.repeat(data.shape[1], 1)[I])) \
            / (theta.repeat(data.shape[1], 1)[I])
        # %%Finish Proportion Calculation
        section[flats] = FLAT_ID_INT
        proportion[flats] = FLAT_ID

        section[section == 8] = 0  # Fence-post error correction
        proportion = (1 + adjust[section]) / 2.0 - adjust[section] * proportion

        return section, proportion

    def _mk_adjacency_matrix(self, section, proportion, flats, elev, mag, dX, dY):
        """
        Calculates the adjacency of connectivity matrix. This matrix tells
        which pixels drain to which.

        For example, the pixel i, will recieve area from np.nonzero(A[i, :])
        at the proportions given in A[i, :]. So, the row gives the pixel
        drain to, and the columns the pixels drained from.
        """
        shp = section.shape
        mat_data = np.row_stack((proportion, 1 - proportion))
        NN = np.prod(shp)
        i12 = np.arange(NN).reshape(shp)
        j1 = - np.ones_like(i12)
        j2 = - np.ones_like(i12)

        # make the connectivity for the non-flats/pits
        j1, j2 = self._mk_connectivity(section, i12, j1, j2)
        j = np.row_stack((j1, j2))
        i = np.row_stack((i12, i12))

        # connectivity for flats/pits
        if self.drain_pits:
            pit_i, pit_j, pit_prop, flats, mag = \
                self._mk_connectivity_pits(i12, flats, elev, mag, dX, dY)

            j = np.concatenate([j.ravel(), pit_j]).astype('int64')
            i = np.concatenate([i.ravel(), pit_i]).astype('int64')
            mat_data = np.concatenate([mat_data.ravel(), pit_prop])

        elif self.drain_flats:
            j1, j2, mat_data, flat_i, flat_j, flat_prop = \
                self._mk_connectivity_flats(
                    i12, j1, j2, mat_data, flats, elev, mag)

            j = np.concatenate([j.ravel(), flat_j]).astype('int64')
            i = np.concatenate([i.ravel(), flat_j]).astype('int64')
            mat_data = np.concatenate([mat_data.ravel(), flat_prop])



        # This prevents no-data values, remove connections when not present,
        # and makes sure that floating point precision errors do not
        # create circular references where a lower elevation cell drains
        # to a higher elevation cell
        I = ~np.isnan(mat_data) & (j != -1) & (mat_data > 1e-8) \
            & (elev.ravel()[j] <= elev.ravel()[i])

        mat_data = mat_data[I]
        j = j[I]
        i = i[I]

        # %%Make the matrix and initialize
        # What is A? The row i area receives area contributions from the
        # entries in its columns. If all the entries in my columns have
        #  drained, then I can drain.
        A = sps.csc_matrix((mat_data.ravel(),
                            np.row_stack((j.ravel(), i.ravel()))),
                           shape=(NN, NN))
        normalize = np.array(A.sum(0) + 1e-16).squeeze()
        A = np.dot(A, sps.diags(1/normalize, 0))

        return A

    def _mk_connectivity(self, section, i12, j1, j2):
        """
        Helper function for _mk_adjacency_matrix. Calculates the drainage
        neighbors and proportions based on the direction. This deals with
        non-flat regions in the image. In this case, each pixel can only
        drain to either 1 or two neighbors.
        """
        shp = np.array(section.shape) - 1

        facets = self.facets

        for ii, facet in enumerate(facets):
            e1 = facet[1]
            e2 = facet[2]
            I = section[1:-1, 1:-1] == ii
            j1[1:-1, 1:-1][I] = i12[1 + e1[0]:shp[0] + e1[0],
                                    1 + e1[1]:shp[1] + e1[1]][I]
            j2[1:-1, 1:-1][I] = i12[1 + e2[0]:shp[0] + e2[0],
                                    1 + e2[1]:shp[1] + e2[1]][I]

        # Now do the edges
        # left edge
        slc0 = (slice(1, -1), slice(0, 1))
        for ind in [0, 1, 6, 7]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[1 + e1[0]:shp[0] + e1[0], e1[1]][I.ravel()]
            j2[slc0][I] = i12[1 + e2[0]:shp[0] + e2[0], e2[1]][I.ravel()]

        # right edge
        slc0 = (slice(1, -1), slice(-1, None))
        for ind in [2, 3, 4, 5]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[1 + e1[0]:shp[0] + e1[0],
                              shp[1] + e1[1]][I.ravel()]
            j2[slc0][I] = i12[1 + e2[0]:shp[0] + e2[0],
                              shp[1] + e2[1]][I.ravel()]

        # top edge
        slc0 = (slice(0, 1), slice(1, -1))
        for ind in [4, 5, 6, 7]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[e1[0], 1 + e1[1]:shp[1] + e1[1]][I.ravel()]
            j2[slc0][I] = i12[e2[0], 1 + e2[1]:shp[1] + e2[1]][I.ravel()]

        # bottom edge
        slc0 = (slice(-1, None), slice(1, -1))
        for ind in [0, 1, 2, 3]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[shp[0] + e1[0],
                              1 + e1[1]:shp[1] + e1[1]][I.ravel()]
            j2[slc0][I] = i12[shp[0] + e2[0],
                              1 + e2[1]:shp[1] + e2[1]][I.ravel()]

        # top-left corner
        slc0 = (slice(0, 1), slice(0, 1))
        for ind in [6, 7]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[e1[0], e1[1]]
                j2[slc0] = i12[e2[0], e2[1]]

        # top-right corner
        slc0 = (slice(0, 1), slice(-1, None))
        for ind in [4, 5]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[e1[0], shp[1] + e1[1]]
                j2[slc0] = i12[e2[0], shp[1] + e2[1]]

        # bottom-left corner
        slc0 = (slice(-1, None), slice(0, 1))
        for ind in [0, 1]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[shp[0] + e1[0], e1[1]]
                j2[slc0] = i12[shp[0] + e2[0], e2[1]]

        # bottom-right corner
        slc0 = (slice(-1, None), slice(-1, None))
        for ind in [2, 3]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[e1[0] + shp[0], shp[1] + e1[1]]
                j2[slc0] = i12[e2[0] + shp[0], shp[1] + e2[1]]

        return j1, j2

    def _mk_connectivity_pits(self, i12, flats, elev, mag, dX, dY):
        """
        Helper function for _mk_adjacency_matrix. This is a more general
        version of _mk_adjacency_flats which drains pits and flats to nearby
        but non-adjacent pixels. The slope magnitude (and flats mask) is
        updated for these pits and flats so that the TWI can be computed.
        """

        e = elev.ravel()

        pit_i = []
        pit_j = []
        pit_prop = []
        warn_pits = []

        pits = i12[flats & (elev > 0)]
        I = np.argsort(e[pits])
        for pit in pits[I]:
            # find drains
            pit_area = np.array([pit], 'int64')

            drain = None
            epit = e[pit]
            for it in range(self.drain_pits_max_iter):
                border = get_border_index(pit_area, elev.shape, elev.size)

                eborder = e[border]
                emin = eborder.min()
                if emin < epit:
                    drain = border[eborder < epit]
                    break

                pit_area = np.concatenate([pit_area, border[eborder == emin]])

            if drain is None:
                warn_pits.append(pit)
                continue

            ipit, jpit = np.unravel_index(pit, elev.shape)
            Idrain, Jdrain = np.unravel_index(drain, elev.shape)

            # filter by drain distance in coordinate space
            if self.drain_pits_max_dist:
                dij = np.sqrt((ipit - Idrain)**2 + (jpit-Jdrain)**2)
                b = dij <= self.drain_pits_max_dist
                if not b.any():
                    warn_pits.append(pit)
                    continue
                drain = drain[b]
                Idrain = Idrain[b]
                Jdrain = Jdrain[b]

            # calculate real distances
            dx = [_get_dX_mean(dX, ipit, idrain) * (jpit - jdrain)
                  for idrain, jdrain in zip(Idrain, Jdrain)]
            dy = [dY[make_slice(ipit, idrain)].sum() for idrain in Idrain]
            dxy = np.sqrt(np.array(dx)**2 + np.array(dy)**2)

            # filter by drain distance in real space
            if self.drain_pits_max_dist_XY:
                b = dxy <= self.drain_pits_max_dist_XY
                if not b.any():
                    warn_pits.append(pit)
                    continue
                drain = drain[b]
                dxy = dxy[b]

            # calculate magnitudes
            s = (e[pit]-e[drain]) / dxy

            # connectivity info
            # TODO proportion calculation (_mk_connectivity_flats used elev?)
            pit_i += [pit for i in drain]
            pit_j += drain.tolist()
            pit_prop += s.tolist()

            # update pit magnitude and flats mask
            mag[ipit, jpit] = np.mean(s)
            flats[ipit, jpit] = False

        if warn_pits:
            warnings.warn("Warning %d pits had no place to drain to in this "
                          "chunk" % len(warn_pits))

        # Note: returning flats and mag here is not strictly necessary
        return (np.array(pit_i, 'int64'),
                np.array(pit_j, 'int64'),
                np.array(pit_prop, 'float64'),
                flats,
                mag)

    def _mk_connectivity_flats(self, i12, j1, j2, mat_data, flats, elev, mag):
        """
        Helper function for _mk_adjacency_matrix. This calcualtes the
        connectivity for flat regions. Every pixel in the flat will drain
        to a random pixel in the flat. This accumulates all the area in the
        flat region to a single pixel. All that area is then drained from
        that pixel to the surroundings on the flat. If the border of the
        flat has a single pixel with a much lower elevation, all the area will
        go towards that pixel. If the border has pixels with similar elevation,
        then the area will be distributed amongst all the border pixels
        proportional to their elevation.
        """
        nn, mm = flats.shape
        NN = np.prod(flats.shape)
        # Label the flats
        assigned, n_flats = spndi.label(flats, FLATS_KERNEL3)

        flat_ids, flat_coords, flat_labelsf = _get_flat_ids(assigned)
        flat_j = [None] * n_flats
        flat_prop = [None] * n_flats
        flat_i = [None] * n_flats

        # Temporary array to find the flats
        edges = np.zeros_like(flats)
        # %% Calcute the flat drainage
        warn_flats = []
        for ii in range(n_flats):
            ids_flats = flat_ids[flat_coords[ii]:flat_coords[ii+1]]
            edges[:] = 0
            j = ids_flats % mm
            i = ids_flats // mm
            for iii in [-1, 0, 1]:
                for jjj in [-1, 0, 1]:
                    i_2 = i + iii
                    j_2 = j + jjj
                    ids_tmp = (i_2 >= 0) & (j_2 >= 0) & (i_2 < nn) & (j_2 < mm)
                    edges[i_2[ids_tmp], j_2[ids_tmp]] += \
                        FLATS_KERNEL3[iii+1, jjj+1]
            edges.ravel()[ids_flats] = 0
            ids_edge = np.argwhere(edges.ravel()).squeeze()

            flat_elev_loc = elev.ravel()[ids_flats]
            # It is possble for the edges to merge 2 flats, so we need to
            # take the lower elevation to avoid large circular regions
            flat_elev = flat_elev_loc.min()

            loc_elev = elev.ravel()[ids_edge]
            # Filter out any elevations larger than the flat elevation
            # TODO: Figure out if this should be <= or <
            I_filt = loc_elev < flat_elev
            try:
                loc_elev = loc_elev[I_filt]
                loc_slope = mag.ravel()[ids_edge][I_filt]
            except: # If this is fully masked out (i.e. inside a no-data area)
                loc_elev = np.array([])
                loc_slope = np.array([])

            loc_dx = self.dX.mean()

            # Now I have to figure out if I should just use the minimum or
            # distribute amongst many pixels on the flat boundary
            n = len(loc_slope)
            if n == 0:  # Flat does not have anywhere to drain
                # Let's see if the flat goes to the edge. If yes, we'll just
                # distribute the area along the edge.
                ids_flat_on_edge = ((ids_flats % mag.shape[1]) == 0) | \
                    ((ids_flats % mag.shape[1]) == (mag.shape[1] - 1)) | \
                    (ids_flats <= mag.shape[1]) | \
                    (ids_flats >= (mag.shape[1] * (mag.shape[0] - 1)))
                if ids_flat_on_edge.sum() == 0:
                    warn_flats.append(ii)
                    continue

                drain_ids = ids_flats[ids_flat_on_edge]
                loc_proportions = mag.ravel()[ids_flats[ids_flat_on_edge]]
                loc_proportions /= loc_proportions.sum()

                ids_flats = ids_flats[~ids_flat_on_edge]
                # This flat is entirely on the edge of the image
                if len(ids_flats) == 0:
                    # therefore, whatever drains into it is done.
                    continue
                flat_elev_loc = flat_elev_loc[~ids_flat_on_edge]
            else:  # Flat has a place to drain to
                min_edges = np.zeros(loc_slope.shape, bool)
                min_edges[np.argmin(loc_slope)] = True
                # Add to the min edges any edge that is within an error
                # tolerance as small as the minimum
                min_edges = (loc_slope + loc_slope * loc_dx / 2) \
                    >= loc_slope[min_edges]

                drain_ids = ids_edge[I_filt][min_edges]

                loc_proportions = loc_slope[min_edges]
                loc_proportions /= loc_proportions.sum()

            # Now distribute the connectivity amongst the chosen elevations
            # proportional to their slopes

            # First, let all the the ids in the flats drain to 1
            # flat id (for ease)
            one_id = np.zeros(ids_flats.size, bool)
            one_id[np.argmin(flat_elev_loc)] = True

            j1.ravel()[ids_flats[~one_id]] = ids_flats[one_id]
            mat_data.ravel()[ids_flats[~one_id]] = 1
            # Negative indices will be eliminated before making the matix
            j2.ravel()[ids_flats[~one_id]] = -1
            mat_data.ravel()[ids_flats[~one_id] + NN] = 0

            # Now drain the 1 flat to the drains
            j1.ravel()[ids_flats[one_id]] = drain_ids[0]
            mat_data.ravel()[ids_flats[one_id]] = loc_proportions[0]
            if len(drain_ids) > 1:
                j2.ravel()[ids_flats[one_id]] = drain_ids[1]
                mat_data.ravel()[ids_flats[one_id] + NN] = loc_proportions[1]

            if len(loc_proportions > 2):
                flat_j[ii] = drain_ids[2:]
                flat_prop[ii] = loc_proportions[2:]
                flat_i[ii] = np.ones(drain_ids[2:].size, 'int64') * ids_flats[one_id]
        try:
            flat_j = np.concatenate([fj for fj in flat_j if fj is not None])
            flat_prop = \
                np.concatenate([fp for fp in flat_prop if fp is not None])
            flat_i = np.concatenate([fi for fi in flat_i if fi is not None])
        except:
            flat_j = np.array([], 'int64')
            flat_prop = np.array([], 'float64')
            flat_i = np.array([], 'int64')

        if len(warn_flats) > 0:
            warnings.warn("Warning %d flats had no place" % len(warn_flats) +
                          " to drain to --> these are pits (check pit-remove"
                          "algorithm).")
        return j1, j2, mat_data, flat_i, flat_j, flat_prop

    def calc_twi(self):
        """
        Calculates the topographic wetness index and saves the result in
        self.twi.

        Returns
        -------
        twi : array
            Array giving the topographic wetness index at each pixel
        """
        if self.uca is None:
            self.calc_uca()
            gc.collect()  # Just in case
        min_area = self.twi_min_area
        min_slope = self.twi_min_slope
        twi = self.uca.copy()
        if self.apply_twi_limits_on_uca:
            twi[twi > self.uca_saturation_limit * min_area] = \
                self.uca_saturation_limit * min_area
        gc.collect()  # Just in case
        twi = np.log((twi) / (self.mag + min_slope))
        # apply the cap
        if self.apply_twi_limits:
            twi_sat_value = \
                np.log(self.uca_saturation_limit * min_area / min_slope)
            twi[twi > twi_sat_value] = twi_sat_value
        # multiply by 10 for better integer resolution when storing
        self.twi = twi * 10

        gc.collect()  # Just in case
        return twi

    def _plot_connectivity(self, A, data=None, lims=[None, None]):
        """
        A debug function used to plot the adjacency/connectivity matrix.
        This is really just a light wrapper around _plot_connectivity_helper
        """

        if data is None:
            data = self.elev

        B = A.tocoo()
        self._plot_connectivity_helper(B.col, B.row, B.data, data, lims)

    def _plot_connectivity_helper(self, ii, ji, mat_datai, data, lims=[1, 8]):
        """
        A debug function used to plot the adjacency/connectivity matrix.
        """
        from matplotlib.pyplot import quiver, colorbar, clim,  matshow
        I = ~np.isnan(mat_datai) & (ji != -1) & (mat_datai >= 0)
        mat_data = mat_datai[I]
        j = ji[I]
        i = ii[I]
        x = i.astype(float) % data.shape[1]
        y = i.astype(float) // data.shape[1]
        x1 = (j.astype(float) % data.shape[1]).ravel()
        y1 = (j.astype(float) // data.shape[1]).ravel()
        nx = (x1 - x)
        ny = (y1 - y)
        matshow(data, cmap='gist_rainbow'); colorbar(); clim(lims)
        quiver(x, y, nx, ny, mat_data.ravel(), angles='xy', scale_units='xy',
               scale=1, cmap='bone')
        colorbar(); clim([0, 1])

    def _plot_debug_slopes_directions(self):
        """
        A debug function to plot the direction calculated in various ways.
        """
        # %%
        from matplotlib.pyplot import matshow, colorbar, clim, title

        matshow(self.direction / np.pi * 180); colorbar(); clim(0, 360)
        title('Direction')

        mag2, direction2 = self._central_slopes_directions()
        matshow(direction2 / np.pi * 180.0); colorbar(); clim(0, 360)
        title('Direction (central difference)')

        matshow(self.mag); colorbar()
        title('Magnitude')
        matshow(mag2); colorbar(); title("Magnitude (Central difference)")

        # %%
        # Compare to Taudem
        filename = self.file_name
        os.chdir('testtiff')
        try:
            os.remove('test_ang.tif')
            os.remove('test_slp.tif')
        except:
            pass
        cmd = ('dinfflowdir -fel "%s" -ang "%s" -slp "%s"' %
               (os.path.split(filename)[-1], 'test_ang.tif', 'test_slp.tif'))
        taudem._run(cmd)

        td_file = GdalReader(file_name='test_ang.tif')
        td_ang, = td_file.raster_layers
        td_file2 = GdalReader(file_name='test_slp.tif')
        td_mag, = td_file2.raster_layers
        os.chdir('..')

        matshow(td_ang.raster_data / np.pi*180); clim(0, 360); colorbar()
        title('Taudem direction')
        matshow(td_mag.raster_data); colorbar()
        title('Taudem magnitude')

        matshow(self.elev); colorbar()
        title('The test data (elevation)')

        diff = (td_ang.raster_data - self.direction) / np.pi * 180.0
        diff[np.abs(diff) > 300] = np.nan
        matshow(diff); colorbar(); clim([-1, 1])
        title('Taudem direction - calculated Direction')

        # normalize magnitudes
        mag2 = td_mag.raster_data
        mag2 /= np.nanmax(mag2)
        mag = self.mag.copy()
        mag /= np.nanmax(mag)
        matshow(mag - mag2); colorbar()
        title('Taudem magnitude - calculated magnitude')
        del td_file
        del td_file2
        del td_ang
        del td_mag


def _get_flat_ids(assigned):
    """
    This is a helper function to recover the coordinates of regions that have
    been labeled within an image. This function efficiently computes the
    coordinate of all regions and returns the information in a memory-efficient
    manner.

    Parameters
    -----------
    assigned : ndarray[ndim=2, dtype=int]
        The labeled image. For example, the result of calling
        scipy.ndimage.label on a binary image

    Returns
    --------
    I : ndarray[ndim=1, dtype=int]
        Array of 1d coordinate indices of all regions in the image
    region_ids : ndarray[shape=[n_features + 1], dtype=int]
        Indexing array used to separate the coordinates of the different
        regions. For example, region k has xy coordinates of
        xy[region_ids[k]:region_ids[k+1], :]
    labels : ndarray[ndim=1, dtype=int]
        The labels of the regions in the image corresponding to the coordinates
        For example, assigned.ravel()[I[k]] == labels[k]
    """
    # MPU optimization:
    # Let's segment the regions and store in a sparse format
    # First, let's use where once to find all the information we want
    ids_labels = np.arange(len(assigned.ravel()), dtype=np.int64)
    I = ids_labels[assigned.ravel().astype(bool)]
    labels = assigned.ravel()[I]
    # Now sort these arrays by the label to figure out where to segment
    sort_id = np.argsort(labels)
    labels = labels[sort_id]
    I = I[sort_id]
    # this should be of size n_features-1
    region_ids = np.where(labels[1:] - labels[:-1] > 0)[0] + 1
    # This should be of size n_features + 1
    region_ids = np.concatenate(([0], region_ids, [len(labels)]))

    return [I, region_ids, labels]

def _tarboton_slopes_directions(data, dX, dY, facets, ang_adj):
    """
    Calculate the slopes and directions based on the 8 sections from
    Tarboton http://www.neng.usu.edu/cee/faculty/dtarb/96wr03137.pdf
    """
    shp = np.array(data.shape) - 1


    direction = np.full(data.shape, FLAT_ID_INT, 'float64')
    mag = np.full(data.shape, FLAT_ID_INT, 'float64')

    slc0 = (slice(1, -1), slice(1, -1))
    for ind in range(8):

        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(1 + e1[0], shp[0] + e1[0]),
                slice(1 + e1[1], shp[1] + e1[1]))
        slc2 = (slice(1 + e2[0], shp[0] + e2[0]),
                slice(1 + e2[1], shp[1] + e2[1]))

        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp)
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)

    # %%Now do the edges
    # if the edge is lower than the interior, we need to copy the value
    # from the interior (as an approximation)
    ids1 = (direction[:, 1] > np.pi / 2) \
        & (direction[:, 1] < 3 * np.pi / 2)
    direction[ids1, 0] = direction[ids1, 1]
    mag[ids1, 0] = mag[ids1, 1]
    ids1 = (direction[:, -2] < np.pi / 2) \
        | (direction[:, -2] > 3 * np.pi / 2)
    direction[ids1, -1] = direction[ids1, -2]
    mag[ids1, -1] = mag[ids1, -2]
    ids1 = (direction[1, :] > 0) & (direction[1, :] < np.pi)
    direction[0, ids1] = direction[1, ids1]
    mag[0, ids1] = mag[1, ids1]
    ids1 = (direction[-2, :] > np.pi) & (direction[-2, :] < 2 * np.pi)
    direction[-1, ids1] = direction[-2, ids1]
    mag[-1, ids1] = mag[-2, ids1]

    # Now update the edges in case they are higher than the interior (i.e.
    # look at the downstream angle)

    # left edge
    slc0 = (slice(1, -1), slice(0, 1))
    for ind in [0, 1, 6, 7]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(1 + e1[0], shp[0] + e1[0]), slice(e1[1], 1 + e1[1]))
        slc2 = (slice(1 + e2[0], shp[0] + e2[0]), slice(e2[1], 1 + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp)
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # right edge
    slc0 = (slice(1, -1), slice(-1, None))
    for ind in [2, 3, 4, 5]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(1 + e1[0], shp[0] + e1[0]),
                slice(shp[1] + e1[1], shp[1] + 1 + e1[1]))
        slc2 = (slice(1 + e2[0], shp[0] + e2[0]),
                slice(shp[1] + e2[1], shp[1] + 1 + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp)
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # top edge
    slc0 = (slice(0, 1), slice(1, -1))
    for ind in [4, 5, 6, 7]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(e1[0], 1 + e1[0]), slice(1 + e1[1], shp[1] + e1[1]))
        slc2 = (slice(e2[0], 1 + e2[0]), slice(1 + e2[1], shp[1] + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'top')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # bottom edge
    slc0 = (slice(-1, None), slice(1, -1))
    for ind in [0, 1, 2, 3]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(shp[0] + e1[0], shp[0] + 1 + e1[0]),
                slice(1 + e1[1], shp[1] + e1[1]))
        slc2 = (slice(shp[0] + e2[0], shp[0] + 1 + e2[0]),
                slice(1 + e2[1], shp[1] + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'bot')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # top-left corner
    slc0 = (slice(0, 1), slice(0, 1))
    for ind in [6, 7]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(e1[0], 1 + e1[0]),
                slice(e1[1], 1 + e1[1]))
        slc2 = (slice(e2[0], 1 + e2[0]),
                slice(e2[1], 1 + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'top')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # top-right corner
    slc0 = (slice(0, 1), slice(-1, None))
    for ind in [4, 5]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(e1[0], 1 + e1[0]),
                slice(shp[1] + e1[1], shp[1] + 1 + e1[1]))
        slc2 = (slice(e2[0], 1 + e2[0]),
                slice(shp[1] + e2[1], shp[1] + 1 + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'top')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # bottom-left corner
    slc0 = (slice(-1, None), slice(0, 1))
    for ind in [0, 1]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(shp[0] + e1[0], shp[0] + 1 + e1[0]),
                slice(e1[1], 1 + e1[1]))
        slc2 = (slice(shp[0] + e2[0], shp[0] + 1 + e2[0]),
                slice(e2[1], 1 + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'bot')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)
    # bottom-right corner
    slc0 = (slice(-1, None), slice(-1, None))
    for ind in [2, 3]:
        e1 = facets[ind][1]
        e2 = facets[ind][2]
        ang = ang_adj[ind]
        slc1 = (slice(shp[0] + e1[0], shp[0] + 1 + e1[0]),
                slice(shp[1] + e1[1], shp[1] + 1 + e1[1]))
        slc2 = (slice(shp[0] + e2[0], shp[0] + 1 + e2[0]),
                slice(shp[1] + e2[1], shp[1] + 1 + e2[1]))
        d1, d2, theta = _get_d1_d2(dX, dY, ind, e1, e2, shp, 'bot')
        mag, direction = _calc_direction(data, mag, direction, ang, d1, d2,
                                         theta, slc0, slc1, slc2)

    mag[mag > 0] = np.sqrt(mag[mag > 0])

    return mag, direction

def _get_d1_d2(dX, dY, ind, e1, e2, shp, topbot=None):
    """
    This finds the distances along the patch (within the eight neighboring
    pixels around a central pixel) given the difference in x and y coordinates
    of the real image. This is the function that allows real coordinates to be
    used when calculating the magnitude and directions of slopes.
    """
    if topbot == None:
        if ind in [0, 3, 4, 7]:
            d1 = dX[slice((e2[0] + 1) // 2, shp[0] + (e2[0] - 1) // 2)]
            d2 = dY[slice((e2[0] + 1) // 2, shp[0] + (e2[0] - 1) // 2)]
            if d1.size == 0:
                d1 = np.array([dX[0]])
                d2 = np.array([dY[0]])
        else:
            d2 = dX[slice((e1[0] + 1) // 2, shp[0] + (e1[0] - 1) // 2)]
            d1 = dY[slice((e1[0] + 1) // 2, shp[0] + (e1[0] - 1) // 2)]
            if d1.size == 0:
                d2 = dX[0]
                d1 = dY[0]
    elif topbot == 'top':
        if ind in [0, 3, 4, 7]:
            d1, d2 = dX[0], dY[0]
        else:
            d2, d1 = dX[0], dY[0]
    elif topbot == 'bot':
        if ind in [0, 3, 4, 7]:
            d1, d2 = dX[-1], dY[-1]
        else:
            d2, d1 = dX[-1], dY[-1]

    theta = np.arctan2(d2, d1)

    return d1.reshape(d1.size, 1), d2.reshape(d2.size, 1), theta.reshape(theta.size, 1)


# Let's calculate the slopes!
def _calc_direction(data, mag, direction, ang, d1, d2, theta,
                    slc0, slc1, slc2):
    """
    This function gives the magnitude and direction of the slope based on
    Tarboton's D_\\infty method. This is a helper-function to
    _tarboton_slopes_directions
    
    slc0 is the point where the slope is being determined
    slc1 is a slice of the data giving the 'd1' direction
    slc2 is a slice of the data giving the 'd2' dirction
    """

    data0 = data[slc0]
    data1 = data[slc1]
    data2 = data[slc2]

    s1 = (data0 - data1) / d1
    s2 = (data1 - data2) / d2
    s1_2 = s1**2

    sd = (data0 - data2) / np.sqrt(d1**2 + d2**2)
    r = np.arctan2(s2, s1)
    rad2 = s1_2 + s2**2

    # Handle special cases
    # should be on diagonal
    b_s1_lte0 = s1 <= 0
    b_s2_lte0 = s2 <= 0
    b_s1_gt0 = s1 > 0
    b_s2_gt0 = s2 > 0

    I1 = (b_s1_lte0 & b_s2_gt0) | (r > theta)
    if I1.any():
        rad2[I1] = sd[I1] ** 2
        r[I1] = theta.repeat(I1.shape[1], 1)[I1]

    I2 = (b_s1_gt0 & b_s2_lte0) | (r < 0)  # should be on straight section
    if I2.any():
        rad2[I2] = s1_2[I2]
        r[I2] = 0

    I3 = b_s1_lte0 & (b_s2_lte0 | (b_s2_gt0 & (sd <= 0)))  # upslope or flat
    rad2[I3] = -1

    I4 = rad2 > mag[slc0]
    if I4.any():
        mag[slc0][I4] = rad2[I4]
        direction[slc0][I4] = r[I4] * ang[1] + ang[0] * np.pi/2

    return mag, direction

def _get_dX_mean(dX, i1, i2):
    if i1 == i2:
        return dX[min(i1, dX.size-1)]
    else:
        return dX[make_slice(i1, i2)].mean()
