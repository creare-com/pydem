# -*- coding: utf-8 -*-
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

Created on Fri Jun 20 11:26:08 2014

@author: mpu

NOTE: x = lat, y = lon -- THIS IS REVERSED FROM THE USUAL IDEA
"""
import os

import numpy as np
from scipy.ndimage import gaussian_filter

from .utils import mk_geotiff_obj, mk_dx_dy_from_geotif_layer, get_fn_from_coords, save_raster, read_raster
PLOT_TESTCASES = True

#if PLOT_TESTCASES:
#    from matplotlib.pyplot import (figure, subplot, imshow, title,
#                                       colorbar, clim, draw)
NO_DATA_VALUE = -9999


def make_file_names(testnum, NN, testdir='testtiff'):
    if type(NN) == list:
        N = NN[0]
    else:
        N = NN

    base = os.path.join(testdir, 'test_NN%03d_%03d' % (N, testnum))
    return {"elev": base + '_elev.tif', "ang": base + '_ang.tif',
            "fel": base + '_fel.tif', 'uca': base + '_uca.tif'}


def mk_xy_latlon(N):
    raster = np.zeros((N, N))
    if os.path.exists('temp.tif'):
        os.remove('temp.tif')
    r, transform = mk_geotiff_obj(raster, 'temp.tif')
    dx, dy, _, _ = mk_dx_dy_from_geotif_layer(r)
    y = np.concatenate([[0], np.cumsum(dx)])
    x = np.concatenate([[0], np.cumsum(dy)])
    # Center it on zero
    x = x - (x.max() - x.min()) / 2.0
    y = y - (y.max() - y.min()) / 2.0
    scale = np.sqrt(x**2 + y**2).max()
    x = x / scale
    y = y / scale
    Y, X = np.meshgrid(y, x)

    return X, Y


def make_elev_ang(testnum, NN, raster, angle, uca=None, testdir='testtiff'):
    # Convert masked array to filled array
    raster = raster.filled()
    angle = angle.filled()
    filename = make_file_names(testnum, NN, testdir)
    try:
        if not os.path.exists(testdir):
            os.makedirs(testdir)
        if not os.path.exists(filename['elev']):
            elev, transform = mk_geotiff_obj(raster, filename['elev'])
            del transform
            del elev
            print("Created", filename['elev'])
            ang, driver = mk_geotiff_obj(angle, filename['ang'])
            del ang
            del driver
            print("Created", filename['ang'])
        if not os.path.exists(filename['uca']) and uca is not None:
            ucaf, driver = mk_geotiff_obj(uca, filename['uca'])
            del ucaf
            del driver
            print("Created", filename['uca'])
        else:
            print("Already exists or None", filename['uca'])
    finally:
        ucaf = None
        elev = None
        ang = None
        driver = None

# Define the inidividual elevation/angles for the test-cases
def case_cone(x, y, noise=False):
    # %%Cone
    NN = x.shape[0]
    raster = np.ma.masked_array(1 - np.sqrt((y)**2 + (x)**2) / np.sqrt(2.),
                                mask=np.zeros(x.shape, bool),
                                fill_value=NO_DATA_VALUE)
    angle = np.ma.masked_array(np.arctan2(x, -y) + np.pi,
                               mask=np.zeros(x.shape, bool),
                               fill_value=NO_DATA_VALUE)
    ncell = np.round(np.sqrt((y)**2 + (x)**2) / (1.0 / NN))
    uca = np.ma.masked_array(np.pi * ((y)**2 + (x)**2) / ncell * NN**2 / 4.,
                             mask=np.zeros(x.shape, bool),
                             fill_value=NO_DATA_VALUE)
    # We expect the total upstream contributing area to be the
    # total area of the data
    summat = np.zeros_like(uca)
    summat[:, 0] = 1
    summat[:, -1] = 1
    summat[0, :] = 1
    summat[-1, :] = 1
    uca = uca * NN ** 2 / np.nansum(summat * uca)
    if noise:
        np.random.seed(1773)
        raster += np.abs(np.random.randn(*raster.shape)**2) * 0.0003
        raster[:, :] = gaussian_filter(raster, 2, mode='constant')
#
    return raster, angle, uca


def case_cone_scaled(x, y, noise):
    raster, angle, uca = case_cone(x, y, noise)
    raster = raster - raster.ravel().min()  # (make it all positive)
    return raster, angle, uca


def case_line(x, y, line):
    raster = np.ma.masked_array(line[0] * x + line[1] * y,
                                mask=np.zeros(x.shape, bool),
                                fill_value=NO_DATA_VALUE)
    # normalize
    raster = raster - raster.min()
    raster = raster / raster.max()

    angle = np.ma.masked_array(np.arctan2(-line[0], line[1])
                               * np.ones(raster.shape) + np.pi,
                               mask=np.zeros(x.shape, bool),
                               fill_value=NO_DATA_VALUE)
    NN = x.shape[0]

    if line[0] > 0:
        x_line = 1
    else:
        x_line = -1
    if line[1] > 0:
        y_line = 1
    else:
        y_line = -1

    tx = (x_line - x) / (line[0] + 1e-17)
    ty = (y_line - y) / (line[1] + 1e-17)
    if line[0] == 0:
        t = ty
    elif line[1] == 0:
        t = tx
    else:
        t = np.minimum(tx, ty)
    uca = t  # * (line[0]**2 + line[1]**2)** (0.5)
    uca = np.ma.masked_array(np.round(t * NN) / 2,
                             mask=np.zeros(x.shape, bool),
                             fill_value=NO_DATA_VALUE)
    return raster, angle, uca


def case_opposing_slopes(x, y, line):
    raster = np.ma.masked_array(line[0] * x + line[1] * y,
                                mask=np.zeros(x.shape, bool),
                                fill_value=NO_DATA_VALUE)
    I = -x * line[1] + y * line[0] > 0
    raster[I] = -line[0] * x[I] - line[1] * y[I]
    # normalize
    raster = raster - raster.min()
    raster = raster / raster.max()

    angle = np.ma.masked_array(np.arctan2(-line[0], line[1])
                               * np.ones(raster.shape) + np.pi,
                               mask=np.zeros(x.shape, bool),
                               fill_value=NO_DATA_VALUE)
    angle[I] = np.arctan2(line[0], -line[1]) + np.pi
    return raster, angle


def case_ring_flat(x, y, drains):
    raster, angle, uca = case_cone(x, y)
    if type(drains[0]) != list:
        drains = [drains]
    # Set the flat
    I1 = raster >= 0.7
    I = (raster > 0.5) & (raster < 0.7)
    angle[I] = -1

    raster2 = raster.copy()
    for drain in drains:
        raster[drain[0], drain[1]] -= np.min(raster[drain[0], drain[1]])
    raster[I] = 0.51
    raster[I1] = raster2[I1]
    return raster, angle


def case_top_flat(x, y, drains):
    raster, angle, uca = case_cone(x, y)
    if type(drains[0]) != list:
        drains = [drains]
    # Set the flat
    I = raster >= 0.7
    angle[I] = -1

    for drain in drains:
        raster[drain[0], drain[1]] -= np.min(raster[drain[0], drain[1]])
    raster[I] = 0.75
    return raster, angle


def case_line_flat(x, y, line):
    raster, angle, uca = case_line(x, y, line)
    flat_raster = np.ma.masked_array(1 - np.sqrt((y)**2 + (x)**2) / np.sqrt(2.),
                                     mask=np.zeros(x.shape, bool),
                                     fill_value=NO_DATA_VALUE)
    # normalize
    raster = raster - raster.min()
    raster = raster / raster.max()

    # Set the flat
    I = flat_raster >= 0.7
    raster[I] = 0.5
    angle[I] = -1

    return raster, angle


def case_pit_of_dispair(x, y, drains):
    if len(drains) > 0 and type(drains[0]) != list:
        drains = [drains]
        # %% The pit of dispair
    raster = np.ma.masked_array(1 + np.sqrt((y)**2 + (x)**2) / np.sqrt(2.),
                                mask=np.zeros(x.shape, bool),
                                fill_value=NO_DATA_VALUE)

    angle = np.ma.masked_array(np.arctan2(-x, y) + np.pi,
                               mask=np.zeros(x.shape, bool),
                               fill_value=NO_DATA_VALUE)
    for drain in drains:
        N = len(raster[drain[0], drain[1]].ravel())
        raster[drain[0], drain[1]] = np.linspace(0, 1, N)\
            .reshape(raster[drain[0], drain[1]].shape)

    return raster, angle


def case_trough_of_dispair(x, y, line):
    # %% The trough of dispair
    NN = x.shape[0]
    raster = np.ma.masked_array(line[0] * x + line[1] * y,
                                mask=np.zeros(x.shape, bool),
                                fill_value=NO_DATA_VALUE)
    I = x * line[1] + y * line[0] > 0
    raster[I] = -line[0] * x[I] - line[1] * y[I]
    raster += 0.1 * (-line[0] * x + line[1] * y)

    # normalize
    raster = raster - raster.min()
    raster = raster / raster.max()

    angle = np.ma.masked_array(np.arctan2(-line[0] * 0.9, line[1] * 1.1)
                               * np.ones(raster.shape) + np.pi,
                               mask=np.zeros(x.shape, bool),
                               fill_value=NO_DATA_VALUE)
    angle[I] = np.arctan2(line[0] * 1.1, -line[1] * 0.9) + np.pi
    angle[[np.arange(NN), np.arange(NN)]] = 3 * np.pi / 4

    return raster, angle


def case_real_data(x, y, filename='N43W-72_N44W-71_elev.tif', NN=None):
# def case_real_data(x, y, filename='N72W-112N73W-111_elev.tif', NN=None):
    if NN is None:
        NX, NY = x.shape
        NX0, NY0 = [0, 0]
    elif type(NN) == list:
        NX0, NX, NY0, NY = NN
    else:
        NX0, NX, NY0, NY = [0, NN, 0, NN]
    elev_file = read_raster(file_name=filename)
    test_data = elev_file.read()
    raster = test_data[NX0:NX, NY0:NY]
    del elev_file
    del test_data
    angle = -1 + 0 * raster
    return raster, angle


def case_pit_of_carkoon(x, y):
    raster, angle = case_pit_of_dispair(x, y, [])

    I = np.argmin(raster)
    raster.mask.ravel()[I] = True
    angle.mask.ravel()[I] = True
    return raster, angle


def case_sea_of_saw(x, y):
    raster, angle, uca = case_cone(x, y)
    I = (raster < 0.4) & (x > 0)
    raster.mask[I] = True
    angle.mask[I] = True
    return raster, angle


def spiral(x, y):
    n, m = x.shape
    # make the walls
    raster = np.ma.masked_array((1.0 - np.maximum(np.abs(x), np.abs(y))**2)
                                * (n * m) / 2 + (n + m) * 2,
                                mask=np.zeros(x.shape, bool),
                                fill_value=NO_DATA_VALUE)

    # make the spiral
    start = [0, 1]
    stop = [n - 2, m - 2]
    i, j = start
    start = [-1, 1]
    elev = 0
    count = 0
    sgn = [1, 1]
    while count < n*m / 2:
        count += 1
        raster[i, j] = elev
        elev += 1
        if i < stop[0] and sgn[0] > 0:
            i += 1
        elif j < stop[1] and sgn[1] > 0:
            j += 1
        elif i == stop[0] and j == stop[1] and sgn[1] > 0:
            start[0] += 2
            start[1] += 2
            sgn = [-1, -1]
            i -= 1
        elif i > start[0] and sgn[0] < 0:
            i -= 1
        elif j > start[1] and sgn[1] < 0:
            j -= 1
        elif i == start[0] and j == start[1] and sgn[1] < 0:
            stop[0] -= 2
            stop[1] -= 2
            sgn = [1, 1]
            i += 1
        else:
            break

    return raster, raster * 0


def mk_test_multifile(testnum, NN, testdir, nx_grid=3, ny_grid=4, nx_overlap=16,
                      ny_overlap=32, lat=[46, 45], lon=[-73, -72]):
    """
    Written to make test case for multi-file edge resolution.
    """
    path = os.path.split(make_file_names(testnum, NN, os.path.join(
                                               testdir, 'chunks'))['elev'])[0]
    try:
        os.makedirs(path)
    except:
        pass

    def _get_chunk_edges(NN, chunk_size, chunk_overlap):
        chunk_size = int(chunk_size)
        left_edge = np.arange(0, NN - chunk_overlap, chunk_size)
        left_edge[1:] -= chunk_overlap // 2
        right_edge = np.arange(0, NN - chunk_overlap, chunk_size)
        right_edge[:-1] = right_edge[1:] + int(np.ceil(chunk_overlap / 2))
        right_edge[-1] = NN
        right_edge = np.minimum(right_edge, NN)
        return left_edge, right_edge

    elev_data, ang_data, fel_data = get_test_data(testnum, NN)
    raster = elev_data

    ni, nj = raster.shape

    top_edge, bottom_edge = _get_chunk_edges(ni, np.ceil(ni / ny_grid), ny_overlap)
    left_edge, right_edge = _get_chunk_edges(nj, np.ceil(nj / nx_grid), nx_overlap)

#    gc = elev_data.grid_coordinates
#    lat = gc.y_axis
#    lon = gc.x_axis
    lat = np.linspace(lat[0], lat[1], ni)
    lon = np.linspace(lon[0], lon[1], nj)
    count = 0
    for te, be in zip(top_edge, bottom_edge):
        for le, re in zip(left_edge, right_edge):
            count += 1
            fn = os.path.join(path,
                              get_fn_from_coords((lat[be-1], lon[le], lat[te],
                                                  lon[re-1]), 'elev'))
            print(count, ": [%d:%d, %d:%d]" % (te, be, le, re), \
                '(lat, lon) = (%g to %g, %g to %g)' % (lat[te], lat[be-1],
                                                       lon[le], lon[re-1]), \
                'min,max = (%g to %g, %g to %g)' % (lat.min(), lat.max(),
                                                    lon.min(), lon.max()))
            mk_geotiff_obj(raster[te:be, le:re], fn,
                           bands=1,
                           lat=[lat[te], lat[be-1]], lon=[lon[le], lon[re-1]])


# %% MAKE ALL THE TESTS
def make_test_files(NN=32, plotflag=False, testdir='testtiff', testnum=None):
    # Common
    if plotflag:
        from matplotlib.pyplot import (figure, subplot, imshow, title,
                                       colorbar, clim, draw)

    if type(NN) == list:
        N = np.max(NN)
    else:
        N = NN
    x, y = np.mgrid[-1:1:complex(0, N), -1:1:complex(0, N)]
    lat, lon = mk_xy_latlon(N)

    # The first brace it just to avoid the line-continuouation character
    tests = [[case_cone]  # case 0
             + [lambda x, y: case_line(x, y, [1, 0]),
                lambda x, y: case_line(x, y, [-1, 0]),
                lambda x, y: case_line(x, y, [0, 1]),
                lambda x, y: case_line(x, y, [0, -1]),
                lambda x, y: case_line(x, y, [1, 1]),
                lambda x, y: case_line(x, y, [-1, 1]),
                lambda x, y: case_line(x, y, [1, -1]),
                lambda x, y: case_line(x, y, [-1, -1])
                ]  # cases 1-8, 8 cases in total for the lines
             + [lambda x, y: case_opposing_slopes(x, y, [1, 0]),
                lambda x, y: case_opposing_slopes(x, y, [0, 1]),
                lambda x, y: case_opposing_slopes(x, y, [1, 1]),
                lambda x, y: case_opposing_slopes(x, y, [-1, 1])
                ]  # cases 9-12, 4 cases in total for the opposing slopes
             + [lambda x, y: case_ring_flat(x, y, [slice(NN), slice(NN)]),
                lambda x, y: case_ring_flat(x, y, [slice(NN//2, NN//2+1),
                                                   slice(NN//2, NN)]),
                lambda x, y: case_ring_flat(x, y, [[slice(NN//2, NN//2+1),
                                                    slice(0, NN//2)],
                                                   [slice(NN//2, NN//2+1),
                                                    slice(NN//2, NN)]]),
                lambda x, y: case_ring_flat(x, y, [[slice(NN//2, NN),
                                                    slice(NN//2, NN//2+1)],
                                                   [slice(NN//2, NN//2+1),
                                                    slice(0, NN//2)],
                                                   [slice(NN//2, NN//2+1),
                                                    slice(NN//2, NN)]]),
                lambda x, y: case_ring_flat(x, y, [[slice(0, NN//2),
                                                    slice(NN//2, NN//2+1)],
                                                   [slice(NN//2, NN),
                                                    slice(NN//2, NN//2+1)],
                                                   [slice(NN//2, NN//2+1),
                                                    slice(0, NN//2)],
                                                   [slice(NN//2, NN//2+1),
                                                    slice(NN//2, NN)]])
                ]  # cases 13-17, 5 cases in total for the ring flats
             + [lambda x, y: case_top_flat(x, y, [slice(NN), slice(NN)]),
                lambda x, y: case_top_flat(x, y, [slice(NN//2, NN//2+1),
                                                  slice(NN//2, NN)]),
                lambda x, y: case_top_flat(x, y, [slice(NN//2, NN//2+1),
                                                  slice(0, NN//2)]),
                lambda x, y: case_top_flat(x, y, [slice(NN//2, NN),
                                                  slice(NN//2, NN//2+1)]),
                lambda x, y: case_top_flat(x, y, [slice(0, NN//2),
                                                  slice(NN//2, NN//2+1)])
                ]  # cases 18-22, 5 cases total for top flats
             + [lambda x, y: case_line_flat(x, y, [-1, -1])]  # case 23
             + [lambda x, y: case_pit_of_dispair(x, y, [slice(NN//2, NN//2+1),
                                                        slice(0, NN//2)]),
                lambda x, y: case_pit_of_dispair(x, y, [slice(0, NN//2),
                                                        slice(NN//2, NN//2+1)])
                ]  # cases 24 and 25
             + [lambda x, y: case_trough_of_dispair(x, y, [-1, 1])]  # case 26
             + [lambda x, y: case_real_data(x, y, NN=NN)]  # case 27
             + [case_pit_of_carkoon]  # Case 28
             + [case_sea_of_saw]  # case 29
             + [spiral]  # case 30
             + [lambda x, y: case_cone(x, y, True)]  # case 31
             + [lambda x, y: case_cone_scaled(lon, lat, True)]  # case 32
             + [lambda x, y: case_cone(x, y, False)]  # case 32
             + [lambda x, y: case_cone_scaled(lon, lat, False)]  # case 33
             ]

    tests = tests[0]  # Remove the outer list
    if testnum is None:
        testnum = 0
    else:
        tests = [tests[testnum]]

    for test in tests:
        res = test(x, y)
        raster, angle = res[:2]
        if len(res) == 3:
            uca = res[2]
        else:
            uca = None
        make_elev_ang(testnum, NN, raster, angle, uca, testdir)

        if plotflag:
            figure(testnum)
            subplot(121)
            imshow(raster, interpolation='none'); colorbar()
            draw()
            title('Elevation %d' % testnum)
            subplot(122)
            imshow(angle / np.pi * 180, interpolation='none'); colorbar(); clim([0, 360])
            draw()
            title('Angle %d' % testnum)
        testnum += 1


def get_test_data(test_num, NN, testdir='testtiff'):
    filenames = make_file_names(test_num, NN, testdir)
    if not os.path.exists(filenames['elev']):
        make_test_files(NN, testdir=testdir, testnum=test_num)
    elev_data = read_raster(filenames['elev']).read(1)
    ang_data = read_raster(filenames['ang']).read(1)
    return elev_data, ang_data, None

