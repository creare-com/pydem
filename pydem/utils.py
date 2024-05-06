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


Helper Utilities
=================
The module supplies helper utility functions that may have some more general
use.


Developer Notes
----------------

Created on Tue Oct 14 17:25:50 2014

@author: mpu
"""
from geopy.distance import distance
import os
import re

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.ndimage import center_of_mass
import rasterio

def read_raster(fn):
    return rasterio.open(fn)

def dem_processor_from_raster_kwargs(fn):
    dataset = read_raster(fn)
    dx, dy, dx2, dy2 = mk_dx_dy_from_geotif_layer(dataset)
    elev = dataset.read(1)
    return dict(dX=dx, dY=dy, elev=elev, bounds=dataset.bounds, transform=dataset.transform,
                dX2=dx2, dY2=dy2)

def mk_transform(lat_top, lon_left, dlat, dlon, lat_lon_centered=False):
    if lat_lon_centered:
        lat_top -= dlat / 2  # expect dlat < 0
        lon_left -= dlon / 2
    transform = rasterio.transform.Affine.translation(lon_left, lat_top) \
            * rasterio.transform.Affine.scale(dlon, dlat)
    return transform

def save_raster(fn, data, crs, transform=None, affine=None):
    if transform is not None:
        transform = rasterio.Affine.from_gdal(*transform)
    elif affine is not None:
        transform = affine

    kwargs2 = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        mode="w",
    )
    with rasterio.open(fn, **kwargs2) as dst:
        r = dst.write(data, 1)
    return rasterio.open(fn, mode='r')

def get_fn(elev, name=None):
    """
    Determines the standard filename for a given GeoTIFF Layer.

    Parameters
    -----------
    elev : GdalReader.raster_layer
        A raster layer from the GdalReader object.
    name : str (optional)
        An optional suffix to the filename.
    Returns
    -------
    fn : str
        The standard <filename>_<name>.tif with suffix (if supplied)
    """
    gcs = elev.grid_coordinates
    coords = [gcs.LLC.lat, gcs.LLC.lon, gcs.URC.lat, gcs.URC.lon]
    return get_fn_from_coords(coords, name)


def get_fn_from_coords(coords, name=None):
    """ Given a set of coordinates, returns the standard filename.

    Parameters
    -----------
    coords : list
        [LLC.lat, LLC.lon, URC.lat, URC.lon]
    name : str (optional)
        An optional suffix to the filename.

    Returns
    -------
    fn : str
        The standard <filename>_<name>.tif with suffix (if supplied)
    """
    NS1 = ["S", "N"][coords[0] > 0]
    EW1 = ["W", "E"][coords[1] > 0]
    NS2 = ["S", "N"][coords[2] > 0]
    EW2 = ["W", "E"][coords[3] > 0]
    new_name = "%s%0.3g%s%0.3g_%s%0.3g%s%0.3g" % \
        (NS1, coords[0], EW1, coords[1], NS2, coords[2], EW2, coords[3])
    if name is not None:
        new_name += '_' + name
    return new_name.replace('.', 'o') + '.tif'


def mk_dx_dy_from_geotif_layer(dataset):
    """
    Extracts the change in x and y coordinates from the geotiff file. Presently
    only supports WGS-84 files.
    """
    if dataset.crs.is_projected:
        dX = np.ones(dataset.shape[0] - 1) * dataset.transform.a
        dX2 = np.ones(dataset.shape[0]) * dataset.transform.a
        dY = np.abs(np.ones(dataset.shape[0] - 1) * dataset.transform.e)
        dY2 = np.abs(np.ones(dataset.shape[0]) * dataset.transform.e)
        return dX, dY, dX2, dY2

    wkt = dataset.crs.to_wkt()
    try:
        ellipsoid = wkt.split('SPHEROID["')[1].split('"')[0].replace(' ', '-')
    except Exception as e:
        print(f"No 'SPHEROID' key in wkt: {e}")

    try:
        ellipsoid = wkt.split('ELLIPSOID["')[1].split('"')[0].replace(' ', '-')
    except Exception as e:
        print(f"No 'ELLIPSOID' key in wkt: {e}")

    d = distance(ellipsoid=ellipsoid)
    dx = dataset.transform.a
    lon = dataset.transform.d + dx / 2
    dy = dataset.transform.e
    lat = dataset.transform.f + dy / 2
    dX = np.zeros((dataset.shape[0] - 1))
    for j in range(len(dX)):
        dX[j] = d.measure((np.clip(lat + dy * (j + 1), -90, 90), lon + dx), (np.clip(lat + dy * (j + 1), -90, 90), lon)) * 1000  # km2m
    dY = np.zeros((dataset.shape[0] - 1))
    for i in range(len(dY)):
        dY[i] = d.measure((np.clip(lat + dy * i, -90, 90), lon), (np.clip(lat + dy * (i + 1), -90, 90), lon)) * 1000  # km2m

    lon = dataset.transform.d + dx
    lat = dataset.transform.f + dy
    dX2 = np.zeros((dataset.shape[0]))
    for j in range(len(dX2)):
        dX2[j] = d.measure((np.clip(lat + dy * (j + 1), -90, 90), lon + dx), (np.clip(lat + dy * (j + 1), -90, 90), lon)) * 1000  # km2m
    dY2 = np.zeros((dataset.shape[0]))
    for i in range(len(dY2)):
        dY2[i] = d.measure((np.clip(lat + dy * i, -90, 90), lon), (np.clip(lat + dy * (i + 1), -90, 90), lon)) * 1000  # km2m

    return dX, dY, dX2, dY2


def mk_geotiff_obj(raster, fn, bands=1, gdal_data_type=np.float32,
                   lat=[46, 45], lon=[-73, -72]):
    """
    Creates a new geotiff file objects using the WGS84 coordinate system, saves
    it to disk, and returns a handle to the python file object and driver

    Parameters
    ------------
    raster : array
        Numpy array of the raster data to be added to the object
    fn : str
        Name of the geotiff file
    bands : int (optional)
        See :py:func:`gdal.GetDriverByName('Gtiff').Create
    gdal_data : gdal.GDT_<type>
        Gdal data type (see gdal.GDT_...)
    lat : list
        northern lat, southern lat
    lon : list
        [western lon, eastern lon]
    """
    NNi, NNj = raster.shape
    pixel_height = -np.abs(lat[0] - lat[1]) / (NNi - 1.0)
    pixel_width = np.abs(lon[0] - lon[1]) / (NNj - 1.0)

    transform = mk_transform(max(lat), min(lon), pixel_height, pixel_width, lat_lon_centered=True)
    r = save_raster(fn, data=raster, crs="EPSG:4326", affine=transform)

    return r, transform


def sortrows(a, i=0, index_out=False, recurse=True):
    """ Sorts array "a" by columns i

    Parameters
    ------------
    a : np.ndarray
        array to be sorted
    i : int (optional)
        column to be sorted by, taken as 0 by default
    index_out : bool (optional)
        return the index I such that a(I) = sortrows(a,i). Default = False
    recurse : bool (optional)
        recursively sort by each of the columns. i.e.
        once column i is sort, we sort the smallest column number
        etc. True by default.

    Returns
    --------
    a : np.ndarray
        The array 'a' sorted in descending order by column i
    I : np.ndarray (optional)
        The index such that a[I, :] = sortrows(a, i). Only return if
        index_out = True

    Examples
    ---------
    >>> a = array([[1,2],[3,1],[2,3]])
    >>> b = sortrows(a,0)
    >>> b
    array([[1, 2],
           [2, 3],
           [3, 1]])
    c, I = sortrows(a,1,True)

    >>> c
    array([[3, 1],
           [1, 2],
           [2, 3]])
    >>> I
    array([1, 0, 2])
    >>> a[I,:] - c
    array([[0, 0],
           [0, 0],
           [0, 0]])
    """
    I = np.argsort(a[:, i])
    a = a[I, :]
    # We recursively call sortrows to make sure it is sorted best by every
    # column
    if recurse & (len(a[0]) > i + 1):
        for b in np.unique(a[:, i]):
            ids = a[:, i] == b
            colids = list(range(i)) + list(range(i+1, len(a[0])))
            a[np.ix_(ids, colids)], I2 = sortrows(a[np.ix_(ids, colids)],
                                                  0, True, True)
            I[ids] = I[np.nonzero(ids)[0][I2]]

    if index_out:
        return a, I
    else:
        return a

def get_adjacent_index(I, shape, size):
    """
    Find indices 2d-adjacent to those in I. Helper function for get_border*.

    Parameters
    ----------
    I : np.ndarray(dtype=int)
        indices in the flattened region
    shape : tuple(int, int)
        region shape
    size : int
        region size (technically computable from shape)

    Returns
    -------
    J : np.ndarray(dtype=int)
        indices orthogonally and diagonally adjacent to I

    """

    m, n = shape
    In = I % n
    bL = In != 0
    bR = In != n-1

    J = np.concatenate([
        # orthonally adjacent
        I - n,
        I[bL] - 1,
        I[bR] + 1,
        I + n,

        # diagonally adjacent
        I[bL] - n-1,
        I[bR] - n+1,
        I[bL] + n-1,
        I[bR] + n+1])

    # remove indices outside the array
    J = J[(J>=0) & (J<size)]

    return J

def get_border_index(I, shape, size):
    """
    Get flattened indices for the border of the region I.

    Parameters
    ----------
    I : np.ndarray(dtype=int)
        indices in the flattened region.
    size : int
        region size (technically computable from shape argument)
    shape : tuple(int, int)
        region shape

    Returns
    -------
    J : np.ndarray(dtype=int)
        indices orthogonally and diagonally bordering I
    """

    J = get_adjacent_index(I, shape, size)

    # instead of setdiff?
    # border = np.zeros(size)
    # border[J] = 1
    # border[I] = 0
    # J, = np.where(border)

    return np.setdiff1d(J, I)

def get_border_mask(region):
    """
    Get border of the region as a boolean array mask.

    Parameters
    ----------
    region : np.ndarray(shape=(m, n), dtype=bool)
        mask of the region

    Returns
    -------
    border : np.ndarray(shape=(m, n), dtype=bool)
        mask of the region border (not including region)
    """

    # common special case (for efficiency)
    internal = region[1:-1, 1:-1]
    if internal.all() and internal.any():
        return ~region

    I, = np.where(region.ravel())
    J = get_adjacent_index(I, region.shape, region.size)

    border = np.zeros(region.size, dtype='bool')
    border[J] = 1
    border[I] = 0
    border = border.reshape(region.shape)

    return border

_ORTH2 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
_SQRT2 = np.sqrt(2.0)
def get_distance(region, src):
    """
    Compute within-region distances from the src pixels.

    Parameters
    ----------
    region : np.ndarray(shape=(m, n), dtype=bool)
        mask of the region
    src : np.ndarray(shape=(m, n), dtype=bool)
        mask of the source pixels to compute distances from.

    Returns
    -------
    d : np.ndarray(shape=(m, n), dtype=float)
        approximate within-region distance from the nearest src pixel;
        (distances outside of the region are arbitrary).
    """

    dmax = float(region.size)
    d = np.full(region.shape, dmax)
    d[src] = 0
    for n in range(region.size):
        d_orth = minimum_filter(d, footprint=_ORTH2) + 1
        d_diag = minimum_filter(d, (3, 3)) + _SQRT2
        d_adj = np.minimum(d_orth[region], d_diag[region])
        d[region] = np.minimum(d_adj, d[region])
        if (d[region] < dmax).all():
            break
    return d

def make_slice(a, b):
    if a < b:
        return slice(a, b)
    else:
        return slice(b, a)

def grow_slice(slc, size):
    """
    Grow a slice object by 1 in each direction without overreaching the list.

    Parameters
    ----------
    slc: slice
        slice object to grow
    size: int
        list length

    Returns
    -------
    slc: slice
       extended slice

    """

    return slice(max(0, slc.start-1), min(size, slc.stop+1))

def grow_obj(obj, shape):
    """
    Grow a 2d object by 1 in all directions without overreaching the array.

    Parameters
    ----------
    obj: tuple(slice, slice)
        Pair of slices (e.g. from scipy.ndimage.measurements.find_objects)
    shape: tuple(int, int)
        2d array shape

    Returns
    -------
    obj: tuple(slice, slice)
        extended slice objects
    """

    return grow_slice(obj[0], shape[0]), grow_slice(obj[1], shape[1])


def find_centroid(region):
    """
    Finds an approximate centroid for a region that is within the region.

    Parameters
    ----------
    region : np.ndarray(shape=(m, n), dtype='bool')
        mask of the region.

    Returns
    -------
    i, j : tuple(int, int)
        2d index within the region nearest the center of mass.
    """

    x, y = center_of_mass(region)
    w = np.argwhere(region)
    i, j = w[np.argmin(np.linalg.norm(w - (x, y), axis=1))]
    return i, j

def plot_fill_flat(roi, out, region, source, drain, dL, dH):
    from matplotlib import pyplot

    plot_detail = roi.size < 500
    cmap = 'Greens'

    pyplot.figure()

    ax = pyplot.subplot(221)
    pyplot.axis('off')
    pyplot.title('unfilled')
    im = pyplot.imshow(roi, interpolation='none')
    im.set_cmap(cmap)
    if plot_detail:
        y, x = np.where(region); pyplot.plot(x, y, 'k.')
        y, x = np.where(source); pyplot.plot(x, y, lw=0, color='k', marker='$H$', ms=12)
        y, x = np.where(drain);   pyplot.plot(x, y, lw=0, color='k', marker='$L$', ms=12)

    pyplot.subplot(222, sharex=ax, sharey=ax)
    pyplot.axis('off')
    pyplot.title('filled')
    im = pyplot.imshow(out, interpolation='none')
    im.set_cmap(cmap)
    if plot_detail:
        for elev in np.unique(out):
            y, x = np.where(out==elev)
            pyplot.plot(x, y, lw=0, color='k', marker='$%.3f$' % elev, ms=20)

    if plot_detail:
        flat = (minimum_filter(out, (3, 3)) >= out) & region
        y, x = np.where(flat); pyplot.plot(x, y, 'r_', ms=24)

        pyplot.subplot(223, sharex=ax, sharey=ax)
        pyplot.axis('off')
        pyplot.title('dL')
        im = pyplot.imshow(roi, interpolation='none')
        im.set_cmap(cmap)
        for d in np.unique(dL):
            if d == region.size: continue
            y, x = np.where(dL==d)
            pyplot.plot(x, y, lw=0, color='k', marker='$%.2f$' % d, ms=24)

        pyplot.subplot(224, sharex=ax, sharey=ax)
        pyplot.axis('off')
        pyplot.title('dH')
        im = pyplot.imshow(roi, interpolation='none')
        im.set_cmap(cmap)
        for d in np.unique(dH):
            if d == region.size: continue
            y, x = np.where(dH==d)
            pyplot.plot(x, y, lw=0, color='k', marker='$%.2f$' % d, ms=24)

    pyplot.tight_layout()

def plot_drain_pit(self, pit, drain, prop, s, elev, area):
    from matplotlib import pyplot

    cmap = 'Greens'

    ipit, jpit = np.unravel_index(pit, elev.shape)
    Idrain, Jdrain = np.unravel_index(drain, elev.shape)
    Iarea, Jarea = np.unravel_index(area, elev.shape)

    imin = max(0, min(ipit, Idrain.min(), Iarea.min())-1)
    imax = min(elev.shape[0], max(ipit, Idrain.max(), Iarea.max()) + 2)
    jmin = max(0, min(jpit, Jdrain.min(), Jarea.min())-1)
    jmax = min(elev.shape[1], max(jpit, Jdrain.max(), Jarea.max()) + 2)
    roi = (slice(imin, imax), slice(jmin, jmax))

    pyplot.figure()

    ax = pyplot.subplot(221);
    pyplot.axis('off')
    im = pyplot.imshow(elev[roi], interpolation='none'); im.set_cmap(cmap)
    pyplot.plot(jpit-jmin, ipit-imin, lw=0, color='k', marker='$P$', ms=10)
    pyplot.plot(Jarea-jmin, Iarea-imin, 'k.')
    pyplot.plot(Jdrain-jmin, Idrain-imin, lw=0, color='k', marker='$D$', ms=10)

    pyplot.subplot(223, sharex=ax, sharey=ax); pyplot.axis('off')
    im = pyplot.imshow(elev[roi], interpolation='none'); im.set_cmap(cmap)
    for i, j, val in zip(Idrain, Jdrain, prop):
        pyplot.plot(j-jmin, i-imin, lw=0, color='k', marker='$%.2f$' % val, ms=16)

    pyplot.subplot(224, sharex=ax, sharey=ax); pyplot.axis('off')
    im = pyplot.imshow(elev[roi], interpolation='none'); im.set_cmap(cmap)
    for i, j, val in zip(Idrain, Jdrain, s):
        pyplot.plot(j-jmin, i-imin, lw=0, color='k', marker='$%.2f$' % val, ms=16)

    pyplot.tight_layout()
