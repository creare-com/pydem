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

Processing Management Module
==========================================

This module manages the processing of multiple DEM tiles to create a seamless
TWI or UCA field. It does this by processing the elevation tiles at least
twice. The first round resolves any UCA within a tile, and the subsequent
rounds resolve the drainage across tile edges.

Usage Notes
-------------
This module depends on properly named elevation GEOTIFF files. To rename
existing elevation files, see :py:func:`utils.rename_files`. These elevation
tiles should have had pits removed.

This module consists of three classes and a helper function. General users
should only be concerned with the ProcessManager class.

This module generates a large amount of temporary storage data and additional
directories. These temporary files can be significantly larger than the
elevation tiles themselves. Ensure that significant disk space is available.

Multiple instances of the ProcessManager can be processing on different threads
concurrently during the first processing stage. During the edge resolution
stage, only one thread will presently continue to run. The other threads should
automatically 'finish' processing, while one thread will continue.

Developer Notes
-----------------
The EdgeFile and TileEdgeFile classes are similar to the Edge and TileEdge
classes in the dem_processing module. The difference is that the data for
these are stored on disk in temporary files.

Development Notes
------------------

TODO: Improve the speed of updating the edge data -- only update edges that
      changed (i.e. keep a timestamp of previous access)

Created on Tue Oct 14 18:09:58 2014

@author: mpu
"""

import os
import traceback
import subprocess
import numpy as np
import cPickle
import scipy.interpolate as spinterp
import gc

from reader.gdal_reader import GdalReader

from dem_processing import DEMProcessor
from utils import parse_fn, sortrows, get_fn_from_coords


def find_neighbors(neighbors, coords, I, source_files, f, sides):
    """Find the tile neighbors based on filenames

    Parameters
    -----------
    neighbors : dict
        Dictionary that stores the neighbors. Format is
        neighbors["source_file_name"]["side"] = "neighbor_source_file_name"
    coords : list
        List of coordinates determined from the filename.
        See :py:func:`utils.parse_fn`
    I : array
        Sort index. Different sorting schemes will speed up when neighbors
        are found
    source_files : list
        List of strings of source file names
    f : callable
        Function that determines if two tiles are neighbors based on their
        coordinates. f(c1, c2) returns True if tiles are neighbors
    sides : list
        List of 2 strings that give the "side" where tiles are neighbors.

    Returns
    -------
    neighbors : dict
        Dictionary of neighbors

    Notes
    -------
    For example, if Tile1 is to the left of Tile2, then
    neighbors['Tile1']['right'] = 'Tile2'
    neighbors['Tile2']['left'] = 'Tile1'
    """
    for i, c1 in enumerate(coords):
        me = source_files[I[i]]
        # If the left neighbor has already been found...
        if neighbors[me][sides[0]] != '':
            continue
        # could try coords[i:] (+ fixes) for speed if it becomes a problem
        for j, c2 in enumerate(coords):
            if f(c1, c2):
                # then tiles are neighbors neighbors
                neigh = source_files[I[j]]
                neighbors[me][sides[0]] = neigh
                neighbors[neigh][sides[1]] = me
                break
    return neighbors


class EdgeFile(object):
    """
    Small helper class that keeps track of data on an edge. It doesn't care
    if it's a top, bottom, left, or right edge. The edge data is loaded from
    a file only when access to the data is needed. A few metrics are kept
    in memory. The file is created when the edge object is instantiated, and it
    is not automatically destroyed.
    """

    fn = None
    slice = None
    coords = None
    save_path = None
    post_fn = None

    fn_coords = None
    fn_data = None
    fn_done = None
    fn_todo = None

    n_done = None
    n_coulddo = None
    percent_done = None
    _subdir = 'edge'

    def __init__(self, fn, slice_, save_path, overwrite=False):
        self.fn = fn
        self.coords = parse_fn(fn)
        self.slice = slice_
        self.save_path = save_path
        self.post_fn = '_' + str(slice_).replace(' ', '').replace(',', '-')

        # When initialized we also have to create and initialize the files
        # that go with this edge
        self.fn_coords = self.get_fn('coords')
        self.fn_data = self.get_fn('data')
        self.fn_done = self.get_fn('done')
        self.fn_todo = self.get_fn('todo')

        # Open the elevation file and strip out the coordinates, then save init
        # data
        elev_file = GdalReader(file_name=fn)
        elev, = elev_file.raster_layers
        gc = elev.grid_coordinates
        del elev_file  # close file
        del elev       # make sure it's closed
        points = np.meshgrid(gc.x_axis, gc.y_axis)
        coordinates = np.column_stack([pts[slice_].ravel() for pts in points])
        # flip xy coordinates for regular grid interpolator
        coordinates = coordinates[:, ::-1]
        init_data = np.zeros(coordinates.shape[0], float)
        init_bool = np.zeros(coordinates.shape[0], bool)

        if not os.path.exists(self.fn_coords) and not overwrite:
            self.save_data(coordinates, 'coords')
        if not os.path.exists(self.fn_data) and not overwrite:
            self.save_data(init_data, 'data')
        if not os.path.exists(self.fn_done) and not overwrite:
            self.save_data(init_bool, 'done')
        if not os.path.exists(self.fn_todo) and not overwrite:
            self.save_data(~init_bool, 'todo')

        self.update_metrics()

    def get_fn(self, name):
        fn = os.path.join(self.save_path, self._subdir,
                          get_fn_from_coords(self.coords, 'edge_'
                                             + name + self.post_fn))
        return fn + '.npy'

    def save_data(self, data, name):
        fn = self.get_fn(name)
        np.save(fn, data)

    def calc_n_done(self, coulddo, done):
        return (coulddo & done).sum()

    def calc_n_coulddo(self, coulddo):
        return coulddo.sum()

    def calc_percent_done(self, coulddo, done):
        return 100.0 * (coulddo & done).sum() / (coulddo.sum() + 1e-16)

    def coulddo(self, todo, data, done):
        return todo & (data > 0)

    def set_data(self, name, interp=None, data=None):
        if data is None and interp is None:
            raise RuntimeError('Either interp or data needs to be specified in'
                               'set_data')
        # assumed that we are providing data on the same tile
        elif data is not None:
            newdata = data[self.slice].squeeze()
            old_data = newdata  # Needed just for the dtype
        # Assume we're providing data from a different tile
        elif interp is not None:
            newdata = interp(self.get_coordinates()).squeeze()
            old_data = self.get(name)
            I = np.isnan(newdata)
            newdata[I] = old_data[I]

        self.save_data(newdata.astype(old_data.dtype), name)
        self.update_metrics()

    def get_coordinates(self):
        return self.get('coords')

    def get(self, name):
        fn = self.get_fn(name)
        data = np.load(fn)
        return data

    def update_metrics(self):
        todo = self.get('todo')
        done = self.get('done')
        data = self.get('data')
        coulddo = self.coulddo(todo, data, done)
        self.n_done = self.calc_n_done(coulddo, done)
        self.n_coulddo = self.calc_n_coulddo(coulddo)
        self.percent_done = self.calc_percent_done(coulddo, done)
        fn = os.path.join(self.save_path, self._subdir,
                          get_fn_from_coords(self.coords,
                                             'edge_metrics' + self.post_fn))
        np.save(fn, np.array([self.n_done, self.n_coulddo, self.percent_done]))
        # clean up
        del todo
        del done
        del data
        del coulddo

    def load_metrics(self):
        fn = os.path.join(self.save_path, self._subdir,
                          get_fn_from_coords(self.coords,
                                             'edge_metrics' + self.post_fn))
        if os.path.exists(fn + '.npy'):
            self.n_done, self.n_coulddo, self.percent_done = \
                np.load(fn + '.npy')
        else:
            self.n_done, self.n_coulddo, self.percent_done = [None] * 3


class TileEdgeFile(object):
    """
    This is mostly a light-weight interface/helper to all the of the edge
    data stored on disk
    """
    neighbors = None
    edges = None
    save_path = None
    percent_done = None
    max_elev = None

    def __init__(self, elev_source_files, save_path):
        self.neighbors = self.find_neighbors(elev_source_files)
        self.save_path = save_path
        self.initialize_edges(save_path)
        self.fill_percent_done()
        self.fill_max_elevations()

    def find_neighbors(self, elev_source_files):
        neighbors = {fn: {'left': '', 'right': '', 'top': '', 'bottom': '',
                          'top-left': '', 'top-right': '',
                          'bottom-right': '', 'bottom-left': ''}
                     for fn in elev_source_files}
        coords = np.array([parse_fn(fn) for fn in elev_source_files])
        # find the left neighbors (and right)
        top = 2
        bot = 0
        left = 1
        right = 3

        # Sort the coordinates to find neighbors faster
        coords1, I = sortrows(coords.copy(), index_out=True, recurse=True)
        f_right = lambda c1, c2: c2[bot] == c1[bot] and c2[top] == c1[top] \
            and c2[right] > c1[right] and c2[left] <= c1[right]
        neighbors = find_neighbors(neighbors, coords1, I, elev_source_files,
                                   f_right, ['right', 'left'])
        # Right takes care of left on a grid (should always be true)

        coords1, I = sortrows(coords.copy(), i=1, index_out=True, recurse=True)
        f_top = lambda c1, c2: c2[left] == c1[left] and c2[right] == c1[right] \
            and c2[top] > c1[top] and c2[bot] <= c1[top]
        neighbors = find_neighbors(neighbors, coords1, I, elev_source_files,
                                   f_top, ['top', 'bottom'])

        # Hard part is done. now for convenience, let's find the rest of the
        # neighbors
        for key in neighbors.keys():
            for tb in ['top', 'bottom']:
                for lr in ['left', 'right']:
                    top_neig = neighbors[key][tb]
                    if top_neig != '':
                        neighbors[key]['-'.join([tb, lr])] = \
                            neighbors[top_neig][lr]
                    if neighbors[key]['-'.join([tb, lr])] == '' and \
                            neighbors[key][lr] != '':  # try other option
                        neighbors[key]['-'.join([tb, lr])] = \
                            neighbors[neighbors[key][lr]][tb]

        return neighbors

    def initialize_edges(self, save_path=None):
        if save_path is None:
            save_path = self.save_path

        edges = {fn:
            {'left': EdgeFile(fn, [slice(None), slice(0, 1)], save_path),
             'right': EdgeFile(fn, [slice(None), slice(-1, None)], save_path),
             'top': EdgeFile(fn, [slice(0, 1), slice(None)], save_path),
             'bottom': EdgeFile(fn, [slice(-1, None), slice(None)], save_path)}
            for fn in self.neighbors.keys()}
        self.edges = edges
        return edges

    def fill_max_elevations(self):
        max_elev = {}
        for fn in self.edges.keys():
            elev_file = GdalReader(file_name=fn)
            elev, = elev_file.raster_layers
            max_elev[fn] = np.nanmax(elev.raster_data)
            del elev_file  # close file
            del elev
        self.max_elev = max_elev

    def fill_percent_done(self):
        percent_done = {}
        for key, edge in self.edges.iteritems():
            for key1, ed in edge.iteritems():
                ed.update_metrics()
                # ed.load_metrics()
            percent_done[key] = np.array([edge[key2].percent_done
                                         for key2 in edge.keys()])
            percent_done[key] = percent_done[key].sum() \
                / ((percent_done[key] > 0).sum() + 1e-16)
        self.percent_done = percent_done

    def visualize_neighbors(self, neighbors=None):
        if neighbors is None:
            neighbors = self.neighbors
        import matplotlib.pyplot as plt
        coords = np.array([parse_fn(key) for key in neighbors.keys()])
        n_coords = [np.array([parse_fn(neighbors[key][side])
                    for key in neighbors.keys()])
                    for side in ['left', 'right', 'top', 'bottom',
                    'top-right', 'top-left', 'bottom-right', 'bottom-left']]

        top = 2
        bot = 0
        left = 1
        right = 3
        x = (coords[:, left] + coords[:, right]) / 2.0
        y = (coords[:, top] + coords[:, bot]) / 2.0

        n_x = [(n_coord[:, left] + n_coord[:, right]) / 2.0 - x
               for n_coord in n_coords]
        n_y = [(n_coord[:, top] + n_coord[:, bot]) / 2.0 - y
               for n_coord in n_coords]

        self.fill_percent_done()
        colors = np.array([self.percent_done[key] for key in neighbors.keys()])

        plt.scatter(x, y, c=colors, s=400, cmap='CMRmap_r')
        plt.clim(0, 100)
        plt.colorbar()
        for coord in coords:
            plt.plot([coord[left], coord[right], coord[right],
                      coord[left], coord[left]],
                     [coord[top], coord[top], coord[bot],
                      coord[bot], coord[top]])
        for nx, ny in zip(n_x, n_y):
            plt.quiver(x, y, nx, ny, angles='xy', scale_units='xy',
                       scale=1, width=0.005)
        plt.xlim(coords[:, left].min(), coords[:, right].max())
        plt.ylim(coords[:, bot].min(), coords[:, top].max())
        count = 0
        for key, edge in self.edges.iteritems():
            for side in ['left', 'right', 'top', 'bottom']:
                ed = edge[side]
                coordinates = ed.get_coordinates()
                todo = ed.get('todo')
                done = ed.get('done')
                data = (ed.get('data') > 0)
                y, x = coordinates.T
                if side in ['left', 'top']:
                    plt.plot(x[todo & done & data], y[todo & done & data],
                             'bo', mec='b', mfc='none', mew=1,
                             label='could do (left/top)')
                    plt.plot(x[todo & ~done], y[todo & ~done], 'xr',
                             label='not done, could not do (left/top)')
                    plt.plot(x[~todo & done], y[~todo & done], '<g', mec='g',
                             label='done (left/top)')
                else:
                    plt.plot(x[todo & done & data], y[todo & done & data],
                             'bs', mec='b', mfc='none', mew=1,
                             label='could do (right/bot)')
                    plt.plot(x[todo & ~done], y[todo & ~done], '+r',
                             label='not done, could not do (right/bot)')
                    plt.plot(x[~todo & done], y[~todo & done], '>g', mec='g',
                             label='done (right/bot)')

                if count == 0:
                    plt.plot(x[0], y[0], 'bs', mec='b', mfc='none', mew=1,
                             label='could do (right/bot)')
                    plt.plot(x[0], y[0], '+r',
                             label='not done, could not do (right/bot)')
                    plt.plot(x[0], y[0], '>g', mec='g',
                             label='done (right/bot)')
                    plt.legend(loc=0)
                count += 1

                # clean up
                del coordinates
                del todo
                del done
                del data

    def build_interpolator(self, dem_proc):
        # Build an interpolator
        gc = dem_proc.elev.grid_coordinates
#       points = np.meshgrid(gc.x_axis, gc.y_axis)
#       points = np.column_stack([pts.ravel() for pts in points])
#       interp = spinterp.NearestNDInterpolator(points, dem_proc.data.ravel())
#       interp = spinterp.LinearNDInterpolator(points, np.ravel(dem_proc.data),
#                                               fill_value=np.nan)
        interp = spinterp.interpolate.RegularGridInterpolator(
            points=(gc.y_axis[::-1], gc.x_axis),
            values=dem_proc.data[::-1, :].astype(float),
            method='nearest', fill_value=np.nan, bounds_error=False)
        return interp

    def set_neighbor_data(self, elev_fn, dem_proc, interp=None):
        """
        From the elevation filename, we can figure out and load the data and
        done arrays.
        """
        if interp is None:
            interp = self.build_interpolator(dem_proc)
        opp = {'top': 'bottom', 'left': 'right'}
        for key in self.neighbors[elev_fn].keys():
            tile = self.neighbors[elev_fn][key]
            if tile == '':
                continue
            oppkey = key
            for me, neigh in opp.iteritems():
                if me in key:
                    oppkey = oppkey.replace(me, neigh)
                else:
                    oppkey = oppkey.replace(neigh, me)
            opp_edge = self.neighbors[tile][oppkey]
            if opp_edge == '':
                continue

            interp.values = dem_proc.uca[::-1, :]
#            interp.values[:, 0] = np.ravel(dem_proc.uca)  # for other interp.
            # for the top-left tile we have to set the bottom and right edges
            # of that tile, so two edges for those tiles
            for key_ed in oppkey.split('-'):
                self.edges[tile][key_ed].set_data('data', interp)

            interp.values = dem_proc.edge_done[::-1, :].astype(float)
#            interp.values[:, 0] = np.ravel(dem_proc.edge_done)
            for key_ed in oppkey.split('-'):
                self.edges[tile][key_ed].set_data('done', interp)

    def update_edge_todo(self, elev_fn, dem_proc):
        """
        Can figure out how to update the todo based on the elev filename
        """
        for key in self.edges[elev_fn].keys():
            self.edges[elev_fn][key].set_data('todo', data=dem_proc.edge_todo)

    def update_edges(self, elev_fn, dem_proc):
        """
        After finishing a calculation, this will update the neighbors and the
        todo for that tile
        """
        interp = self.build_interpolator(dem_proc)
        self.update_edge_todo(elev_fn, dem_proc)
        self.set_neighbor_data(elev_fn, dem_proc, interp)

    def get_edge_init_data(self, fn, save_path=None):
        """
        Creates the initialization data from the edge structure
        """

        edge_init_data = {key: self.edges[fn][key].get('data') for key in
                          self.edges[fn].keys()}
        edge_init_done = {key: self.edges[fn][key].get('done') for key in
                          self.edges[fn].keys()}
        edge_init_todo = {key: self.edges[fn][key].get('todo') for key in
                          self.edges[fn].keys()}
        return edge_init_data, edge_init_done, edge_init_todo

    def find_best_candidate(self, elev_source_files=None):
        """
        Heuristically determines which tile should be recalculated based on
        updated edge information. Presently does not check if that tile is
        locked, which could lead to a parallel thread closing while one thread
        continues to process tiles.
        """
        self.fill_percent_done()
        i_b = np.argmax(self.percent_done.values())
        if self.percent_done.values()[i_b] <= 0:
            return None

        # check for ties
        I = np.array(self.percent_done.values()) == \
            self.percent_done.values()[i_b]
        if I.sum() == 1:
            pass  # no ties
        else:
            I2 = np.argmax(np.array(self.max_elev.values())[I])
            i_b = I.nonzero()[0][I2]

            # Make sure the apples are still apples
            assert(np.array(self.max_elev.keys())[I][I2]
                   == np.array(self.percent_done.keys())[I][I2])

        if elev_source_files is not None:
            fn = self.percent_done.keys()[i_b]
            lckfn = _get_lockfile_name(fn)
            if os.path.exists(lckfn):  # another process is working on it
                # Find a different Candidate
                i_alt = np.argsort(self.percent_done.values())[::-1]
                for i in i_alt:
                    fn = self.percent_done.keys()[i]
                    lckfn = _get_lockfile_name(fn)
                    if not os.path.exists(lckfn):
                        break
            # Get and return the index
            i_b = elev_source_files.index(fn)

        return i_b


class ProcessManager(object):
    """
    This assumes that the elevation has already been processed. That is,
    pits have been removed.
    """
    twi_status = []
    elev_source_files = []
    _INPUT_FILE_TYPES = ["tif", "tiff", "vrt", "hgt", 'flt', 'adf', 'grib',
                         'grib2', 'grb', 'gr1']
    tile_edge = None
    _DEBUG = False

    def __init__(self, source_path='.', save_path='processed_data',
                 clean_tmp=True, use_cache=True, overwrite_cache=False):
        """
        Processes elevation data inputs, and saves conditioned elevation and
        the topographic wetness index.

        Parameters
        -----------
        source_path: list/str (optional)
            Default: current directory. Directory/location of elevation files.
        save_path: str (optional)
            Location where processed files will be saved. Default value is a
            sub-directory to the source file path called 'processed_data'.
        clean_tmp: bool (optional)
            Default: True. When True, some temporary files and directories are
            removed after calculation. If False, these are not removed.
        use_cache: bool (optional)
            Default: True. When True, this function looks in the expected
            temporary file location and uses any intermediate files already
            present
        overwrite_cache: bool (optional)
            Default: False. If this file already exists, it is not overwritten,
            and no calculation is made. If True, any exisiting file will be
            replaced
        """
        self.source_path = source_path
        self.save_path = save_path
        self.overwrite_cache = overwrite_cache
        self.clean_tmp = clean_tmp
        self.elev_source_files = [os.path.join(source_path, fn)
                                  for fn in os.listdir(source_path)
                                  if os.path.splitext(fn)[-1].replace('.', '')
                                  in self._INPUT_FILE_TYPES]
        self.twi_status = ["Unknown" for sf in self.elev_source_files]
        self.custom_status = ["Unknown" for sf in self.elev_source_files]

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        subdirs = ['ang', 'mag', 'uca', 'twi', 'uca_edge_corrected', 'edge']
        for subdir in subdirs:
            if not os.path.isdir(os.path.join(save_path,  subdir)):
                os.makedirs(os.path.join(save_path,  subdir))

    def process_twi(self, index=None, do_edges=False, skip_uca_twi=False):
        """
        Processes the TWI, along with any dependencies (like the slope and UCA)

        Parameters
        -----------
        index : int/slice (optional)
            Default: None - process all tiles in source directory. Otherwise,
            will only process the index/indices of the files as listed in
            self.elev_source_files
        do_edges : bool (optional)
            Default False. When false, the UCA will be calculated with
            available edge information if the UCA was not previously computed.
            If the UCA was previously computed and do_edges == False, the UCA
            will not be updated. If do_edges == True, the UCA will also be
            recalculated.
        skip_uca_twi : bool (optional)
            Skips the calculation of the UCA and TWI (only calculates the
            magnitude and direction)
        Notes
        ------
        do_edges = False for the first round of the processing, but it is True
        for the second round.
        """
        if index is not None:
            elev_source_files = [self.elev_source_files[index]]
        else:
            elev_source_files = self.elev_source_files
        for i, esfile in enumerate(elev_source_files):
            try:
                fn, status = self.calculate_twi(esfile,
                                                save_path=self.save_path,
                                                do_edges=do_edges,
                                                skip_uca_twi=skip_uca_twi)
                if index is None:
                    self.twi_status[i] = status
                else:
                    self.twi_status[index] = status
            except:
                lckfn = _get_lockfile_name(esfile)
                try:
                    os.remove(lckfn)
                except:
                    pass
                traceback.print_exc()
                print traceback.format_exc()
                if index is None:
                    self.twi_status[i] = "Error " + traceback.format_exc()
                else:
                    self.twi_status[index] = "Error " + traceback.format_exc()

    def process(self, index=None):
        """
        This will completely process a directory of elevation tiles (as
        supplied in the constructor). Both phases of the calculation, the
        single tile and edge resolution phases are run.

        Parameters
        -----------
        index : int/slice (optional)
            Default None - processes all tiles in a directory. See
            :py:func:`process_twi` for additional options.
        """
        # Round 0 of twi processing, process the magnitude and directions of
        # slopes
        print "Starting slope calculation round"
        self.process_twi(index, do_edges=False, skip_uca_twi=True)

        # Round 1 of twi processing
        print "Starting self-area calculation round"
        self.process_twi(index, do_edges=False)

        # Round 2 of twi processing: edge resolution
        i = self.tile_edge.find_best_candidate(self.elev_source_files)

        print "Starting edge resolution round: ",
        count = 0
        i_old = -1
        same_count = 0
        while i is not None and same_count < 3:
            count += 1
            print '*' * 10
            print count, '(%d -- > %d) .' % (i_old, i)
            # %%
            self.process_twi(i, do_edges=True)
            i_old = i
            i = self.tile_edge.find_best_candidate(self.elev_source_files)
            if i_old == i:
                same_count += 1
            else:
                same_count = 0

        print '*'*79
        print '*******    PROCESSING COMPLETED     *******'
        print '*'*79
        return self

    def calculate_twi(self, esfile, save_path, use_cache=True, do_edges=False,
                      skip_uca_twi=False):
        """
        Calculates twi for supplied elevation file

        Parameters
        -----------
        esfile : str
            Path to elevation file to be processed
        save_path: str
            Root path to location where TWI will be saved. TWI will be saved in
            a subdirectory 'twi'.
        use_cache : bool (optional)
            Default True. If a temporary file exists (from a previous run),
            the cached file will be used. Otherwise, if False, existing files
            will be recomputed
        do_edges : bool (optional)
            See :py:func:`process_twi` for details on this argument.
        skip_uca_twi : bool (optional)
            Skips the calculation of the UCA and TWI (only calculates the
            magnitude and direction)
        """
        if os.path.exists(os.path.join(save_path, 'tile_edge.pkl')) and \
                self.tile_edge is None:
            with open(os.path.join(save_path, 'tile_edge.pkl'), 'r') as fid:
                self.tile_edge = cPickle.load(fid)
        elif self.tile_edge is None:
            self.tile_edge = TileEdgeFile(self.elev_source_files, save_path)
            with open(os.path.join(save_path, 'tile_edge.pkl'), 'wb') as fid:
                cPickle.dump(self.tile_edge, fid)


        status = 'Success'  # optimism
        # Check if file is locked
        lckfn = _get_lockfile_name(esfile)
        coords = parse_fn(esfile)
        fn = get_fn_from_coords(coords, 'twi')
        print '*'*79
        if skip_uca_twi:
            print '*'*10, fn, 'Slope Calculation starting...:', '*'*10
        else:
            print '*'*10, fn, 'TWI Calculation starting...:', '*'*10
        print '*'*79
        if os.path.exists(lckfn):  # another process is working on it
            print fn, 'is locked'
            return fn, "Locked"
        else:  # lock this tile
            fid = file(lckfn, 'w')
            fid.close()

        dem_proc = DEMProcessor(esfile)
        # check if the slope already exists for the file. If yes, we should
        # move on to the next tile without doing anything else
        if skip_uca_twi \
                and os.path.exists(dem_proc.get_full_fn('mag', save_path)
                                   + '.npz') \
                and os.path.exists(dem_proc.get_full_fn('ang', save_path)
                                   + '.npz'):
            print dem_proc.get_full_fn('mag', save_path) + '.npz', 'already exists'
            print dem_proc.get_full_fn('ang', save_path) + '.npz', 'already exists'
            # remove lock file
            os.remove(lckfn)
            return fn, 'Cached: Slope'
        # check if the twi already exists for the file. If not in the edge
        # resolution round, we should move on to the next tile
        if os.path.exists(dem_proc.get_full_fn('twi', save_path)) \
                and (do_edges is False):
            print dem_proc.get_full_fn('twi', save_path), 'already exists'
            # remove lock file
            os.remove(lckfn)
            return fn, 'Cached'

        # only calculate the slopes and direction if they do not exist in cache
        fn_ang = dem_proc.get_full_fn('ang', save_path)
        fn_mag = dem_proc.get_full_fn('mag', save_path)
        if os.path.exists(fn_ang + '.npz') and os.path.exists(fn_mag + '.npz')\
                and not self.overwrite_cache:
            dem_proc.load_direction(fn_ang)
            dem_proc.load_slope(fn_mag)
            dem_proc.find_flats()
        else:
            if os.path.exists(fn_ang + '.npz') and os.path_exists(fn_mag + '.npz')\
                    and self.overwrite_cache:
                os.remove(fn_ang)
                os.remove(fn_mag)
            dem_proc.calc_slopes_directions()
            dem_proc.save_slope(save_path, raw=True)
            dem_proc.save_direction(save_path, raw=True)
        if self._DEBUG:
            dem_proc.save_slope(save_path, as_int=False)
            dem_proc.save_direction(save_path, as_int=False)

        if skip_uca_twi:
            # remove lock file
            os.remove(lckfn)
            return fn, status + ":mag-dir-only"

        fn_uca = dem_proc.get_full_fn('uca', save_path)
        fn_uca_ec = dem_proc.get_full_fn('uca_edge_corrected', save_path)
        fn_twi = dem_proc.get_full_fn('twi', save_path)

        # check if edge structure exists for this tile and initialize
        edge_init_data, edge_init_done, edge_init_todo = \
            self.tile_edge.get_edge_init_data(esfile, save_path)

        # Check if uca data exists (if yes, we are in the
        # edge-resolution round)
        uca_init = None
        if os.path.exists(fn_uca + '.npz'):
            if os.path.exists(fn_uca_ec + '.npz'):
                dem_proc.load_uca(fn_uca_ec)
            else:
                dem_proc.load_uca(fn_uca)
            uca_init = dem_proc.uca

        if do_edges or uca_init is None:
            dem_proc.calc_uca(uca_init=uca_init,
                              edge_init_data=[edge_init_data, edge_init_done,
                                              edge_init_todo])

            if uca_init is None:
                dem_proc.save_uca(save_path, raw=True)
                if self._DEBUG:
                    # Also save a geotiff for debugging
                    dem_proc.save_uca(save_path, as_int=False)
            else:
                if os.path.exists(fn_uca_ec):
                    os.remove(fn_uca_ec)
                dem_proc.save_array(dem_proc.uca, None, 'uca_edge_corrected',
                                    save_path, raw=True)
                if self._DEBUG:
                    dem_proc.save_array(dem_proc.uca, None, 'uca_edge_corrected',
                                        save_path, as_int=False)
            # Saving Edge Data, and updating edges
            self.tile_edge.update_edges(esfile, dem_proc)

        dem_proc.calc_twi()
        if os.path.exists(fn_twi):
            os.remove(fn_twi)
        dem_proc.save_twi(save_path, raw=False)

        # clean up for in case
        gc.collect()

        # remove lock file
        os.remove(lckfn)
        # Save last-used dem_proc for debugging purposes
        if self._DEBUG:
            self.dem_proc = dem_proc
        return fn, status

    def process_hillshade(self, index=None):
        def command(esfile, fn):
            cmd = ['gdaldem', 'hillshade', '-s', '111120',
                   '-compute_edges', '-co',  'BIGTIFF=YES', '-of',
                   'GTiff', '-co', 'compress=lzw', '-co', 'TILED=YES',
                   esfile, fn]
            print '<'*8, ' '.join(cmd), '>'*8
            status = subprocess.call(cmd)
            return status
        self.process_command(command, 'hillshade', index)

    def process_command(self, command, save_name='custom', index=None):
        """
        Processes the hillshading

        Parameters
        -----------
        index : int/slice (optional)
            Default: None - process all tiles in source directory. Otherwise,
            will only process the index/indices of the files as listed in
            self.elev_source_files

        """
        if index is not None:
            elev_source_files = [self.elev_source_files[index]]
        else:
            elev_source_files = self.elev_source_files
        save_root = os.path.join(self.save_path, save_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        for i, esfile in enumerate(elev_source_files):
            try:
                status = 'Success'  # optimism
                # Check if file is locked
                lckfn = _get_lockfile_name(esfile)
                coords = parse_fn(esfile)
                fn = get_fn_from_coords(coords, save_name)
                fn = os.path.join(save_root, fn)
                if os.path.exists(lckfn):  # another process is working on it
                    print fn, 'is locked'
                    status = 'locked'
                elif os.path.exists(fn):
                    print fn, 'already exists'
                    status = 'cached'
                else:  # lock this tile
                    print fn, '... calculating ', save_name
                    fid = file(lckfn, 'w')
                    fid.close()

                    # Calculate the custom process for this tile
                    status = command(esfile, fn)

                    os.remove(lckfn)

                if index is None:
                    self.custom_status[i] = status
                else:
                    self.custom_status[index] = status
            except:
                lckfn = _get_lockfile_name(esfile)
                try:
                    os.remove(lckfn)
                except:
                    pass
                traceback.print_exc()
                print traceback.format_exc()
                if index is None:
                    self.custom_status[i] = "Error " + traceback.format_exc()
                else:
                    self.custom_status[index] = "Error " + traceback.format_exc()

def _get_lockfile_name(esfile):
    lckfn = esfile + '.lck'
    return lckfn
