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
"""

from numpy.linalg import inv
from traits.api import *
import numpy as np
import gdal
import gdalconst
import matplotlib, matplotlib.cm

NO_DATA_VALUE = -9999

d_name_to_wkt = {'WGS84' : r'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]',
                 'NAD83' : r'GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.2572221010002,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4269"]]',
                }
d_name_to_epsg = {'WGS84' : 4326,
                  'NAD83': 4269
                 }
d_wkt_to_name = {v:k for k, v in d_name_to_wkt.iteritems()}
d_wkt_to_name[r'GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.2572221010002,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4269"]]'] = 'NAD83'
d_wkt_to_name[r'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'] = 'WGS84'  # afghanistan dem
d_wkt_to_name[r'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS84",6378137,298.2572235604902,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'] = 'WGS84'
d_wkt_to_name[r'GEOGCS["WGS 84",DATUM["unknown",SPHEROID["WGS84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'] = 'WGS84'
d_epsg_to_name = {4326: 'WGS84',
                  4269: 'NAD83',
                 }

# This trait maps a user-friendly name (e.g., WGS84) to an official WKT.
projection_wkt_trait = Trait('WGS84',
             d_name_to_wkt)

class Point(HasStrictTraits):
    """
    Point(Lat, Lon)

    A simple convenience class to deal with latitude and longitude, which
    can be error-prone to convert otherwise.

    Examples
    --------
    >>> Point(-1.426667, -20.583611)
    <Lat,Lon: -1.427, -20.584>

    >>> Point((1, 25, 36), (20, 35, 1), name="Site A")
    <Site A: 1.427, 20.584>

    >>> p = Point(-1.426667, -20.583611)
    >>> p.lat
    -1.426667
    >>> p.lat_dms
    (1.0, 25.0, 36.001199999999599)
    >>> p.lat_dms = (1.0, 2.0, 3.0)
    >>> p
    <Lat,Lon: 1.034, -20.584>

    Conventions
    -----------
    Following ISO 6709,
        * LAT, LON
        * North (lat) and East (lon) are positive.
        * South (lat) and West (lon) are negative.
        * Decimal representation is preferred; Sexagesimal (base-60) is allowed.

    """
    name = Str()
    wkt = projection_wkt_trait

    lon = Float()
    lon_dms = Property(Tuple((Float, Float, Float)))
    def _set_lon_dms(self, dms):
        deg, min, sec = dms
        self.lon = np.sign(deg) * (abs(deg) + min / 60.0 + sec / 3600.0)
    def _get_lon_dms(self):
        deg = np.floor(abs(self.lon))
        min = np.floor((abs(self.lon) - deg) * 60.0)
        sec = np.round((abs(self.lon) - deg - min / 60.0) * 3600.0,
                       4)  # round to 4 decimal places.
        return (np.sign(self.lon) * deg, min, sec)

    lat = Float()
    lat_dms = Property(Tuple((Float, Float, Float)))
    def _set_lat_dms(self, dms):
        deg, min, sec = dms
        self.lat = np.sign(deg) * (abs(deg) + min / 60.0 + sec / 3600.0)
    def _get_lat_dms(self):
        deg = np.floor(abs(self.lat))
        min = np.floor((abs(self.lat) - deg) * 60.0)
        sec = np.round((abs(self.lat) - deg - min / 60.0) * 3600.0,
                       4)  # round to 4 decimal places.
        return (np.sign(self.lat) * deg, min, sec)

    def to_wkt(self, target_wkt):
        # If we're going from WGS84 -> Spherical Mercator, use PyProj, because
        #  there seems to be a bug in OGR that gives us an offset. (GDAL
        #  does fine, though.
        if target_wkt == self.wkt:
            return self
        import osr
        dstSpatialRef = osr.SpatialReference()
        dstSpatialRef.ImportFromEPSG(d_name_to_epsg[target_wkt])
        # dstSpatialRef.ImportFromWkt(d_name_to_wkt[target_wkt])
        srcSpatialRef = osr.SpatialReference()
        srcSpatialRef.ImportFromEPSG(d_name_to_epsg[self.wkt])
        # srcSpatialRef.ImportFromWkt(self.wkt_)
        coordTransform = osr.CoordinateTransformation(srcSpatialRef, dstSpatialRef)
        a, b, c = coordTransform.TransformPoint(self.lon, self.lat)
        return Point(b, a, wkt=target_wkt)

    def __str__(self):
        return "<%s (%s): %02.3f, %03.3f>" % (self.name if self.name else 'Lat,Lon',
                                              self.wkt,
                                         self.lat, self.lon)

    def __repr__(self):
        s =  ("Point(%02.3f, %02.3f, wkt='%s'" %
              (self.lat, self.lon, self.wkt))
        if self.name:
            s += ", name='%s'" % self.name
        s += ")"

        return s

    def __init__(self, lat=None, lon=None, **kwargs):
        HasStrictTraits.__init__(self, **kwargs)
        if lon is not None:
            try:
                self.lon = lon  # float
            except:
                self.lon_dms = lon  # tuple
        if lat is not None:
            try:
                self.lat = lat  # float
            except:
                self.lat_dms = lat  # tuple


def grid_coords_from_corners(upper_left_corner, lower_right_corner, size):
    ''' Points are the outer edges of the UL and LR pixels. Size is rows, columns.
    GC projection type is taken from Points. '''
    assert upper_left_corner.wkt == lower_right_corner.wkt
    geotransform = np.array([upper_left_corner.lon, -(upper_left_corner.lon - lower_right_corner.lon) / float(size[1]), 0,
                            upper_left_corner.lat, 0, -(upper_left_corner.lat - lower_right_corner.lat) / float(size[0])])
    return GridCoordinates(geotransform=geotransform,
                               wkt=upper_left_corner.wkt,
                               y_size=size[0],
                               x_size=size[1])

class GridCoordinates(HasStrictTraits):
    """
    Defines mapping of input layers to real-world time and space.
    """
    date = Date()
    time = Time()
    geotransform = Array('float', [6],)
    wkt = projection_wkt_trait
    x_size = Int()
    y_size = Int()
    x_axis = Property(Array(), depends_on='geotransform, x_size')
    y_axis = Property(Array(), depends_on='geotransform, y_size')

    ULC = Property(Instance(Point), depends_on='x_axis, y_axis')
    URC = Property(Instance(Point), depends_on='x_axis, y_axis')
    LLC = Property(Instance(Point), depends_on='x_axis, y_axis')
    LRC = Property(Instance(Point), depends_on='x_axis, y_axis')
    def _get_ULC(self): return Point(self.geotransform[3], self.geotransform[0], wkt=self.wkt)
    def _get_URC(self): return Point(self.geotransform[3], self.geotransform[0] + self.geotransform[1] * self.x_size, wkt=self.wkt)
    def _get_LLC(self): return Point(self.geotransform[3] + self.geotransform[5] * self.y_size, self.geotransform[0], wkt=self.wkt)
    def _get_LRC(self): return Point(self.geotransform[3] + self.geotransform[5] * self.y_size, self.geotransform[0] + self.geotransform[1] * self.x_size, wkt=self.wkt)

    projection_wkt = Property(Str)  # For backwards compatibility w/ previously pickled layers.
    def _set_projection_wkt(self, val):
        self.wkt = data.layers.types.d_wkt_to_name[val]

    def __repr__(self):
        return '<GridCoordinates: %s -> %s, %d x %d>' % (
                         self.ULC, self.LRC, self.y_size, self.x_size)

    def intersects(self, other_grid_coordinates):
        """ returns True if the GC's overlap. """
        ogc = other_grid_coordinates  # alias
        # for explanation: http://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other
        # Note the flipped y-coord in this coord system.
        ax1, ay1, ax2, ay2 = self.ULC.lon, self.ULC.lat, self.LRC.lon, self.LRC.lat
        bx1, by1, bx2, by2 = ogc.ULC.lon, ogc.ULC.lat, ogc.LRC.lon, ogc.LRC.lat
        if ((ax1 <= bx2) and (ax2 >= bx1) and (ay1 >= by2) and (ay2 <= by1)):
            return True
        else:
            return False



    def unique_str(self):
        """ A string that (ideally) uniquely represents this GC object. This
        helps with naming files for caching. 'Unique' is defined as 'If
        GC1 != GC2, then GC1.unique_str() != GC2.unique_str()'; conversely,
        'If GC1 == GC2, then GC1.unique_str() == GC2.unique_str()'.

        The string should be filename-safe (no \/:*?"<>|).

        ..note::Because of length/readability restrictions, this fxn ignores
        wkt.

        Example output:
        "-180.000_0.250_0.000_90.000_0.000_-0.251_512_612_2013-05-21_12_32_52.945000"
        """

        unique_str = "_".join(["%.3f" % f for f in self.geotransform] +
                              ["%d" % d for d in self.x_size, self.y_size]
                              )
        if self.date is not None:
            unique_str += '_' + str(self.date)
        if self.time is not None:
            unique_str += '_' + str(self.time)
        return unique_str.replace(':', '_')

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and np.allclose(self.geotransform, other.geotransform)
            and (self.x_size == other.x_size)
            and (self.y_size == other.y_size)
            and (self.date == other.date)
            and (self.wkt == other.wkt)
            and (self.time == other.time)
            )


    def __ne__(self, other):
        return not self.__eq__(other)

    @cached_property
    def _get_x_axis(self):
        """See http://www.gdal.org/gdal_datamodel.html for details."""
        # 0,0 is top/left top top/left pixel. Actual x/y coord of that pixel are (.5,.5).
        x_centers = np.linspace(.5, self.x_size - .5, self.x_size)
        y_centers = x_centers * 0
        return (self.geotransform[0]
                + self.geotransform[1] * x_centers
                + self.geotransform[2] * y_centers)

    @cached_property
    def _get_y_axis(self):
        """See http://www.gdal.org/gdal_datamodel.html for details."""
        # 0,0 is top/left top top/left pixel. Actual x/y coord of that pixel are (.5,.5).
        y_centers = np.linspace(.5, self.y_size - .5, self.y_size)
        x_centers = y_centers * 0
        return (self.geotransform[3]
                + self.geotransform[4] * x_centers
                + self.geotransform[5] * y_centers)



    def raster_to_projection_coords(self, pixel_x, pixel_y):
        """ Use pixel centers when appropriate.
        See documentation for the GDAL function GetGeoTransform for details. """
        h_px_py = np.array([1, pixel_x, pixel_y])
        gt = np.array([[1, 0, 0], self.geotransform[0:3], self.geotransform[3:6]])
        arr = np.inner(gt, h_px_py)
        return arr[2], arr[1]

    def projection_to_raster_coords(self, lat, lon):
        """ Returns pixel centers.
        See documentation for the GDAL function GetGeoTransform for details. """
        r_px_py = np.array([1, lon, lat])
        tg = inv(np.array([[1, 0, 0], self.geotransform[0:3], self.geotransform[3:6]]))
        return np.inner(tg, r_px_py)[1:]

    def _as_gdal_dataset(self, driver="MEM", n_raster_count=1,
                   file_name="memory.tif", data_type=gdalconst.GDT_Float32
                   ):
        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(file_name, int(self.x_size), int(self.y_size),
                                n_raster_count, data_type)

        dataset.SetGeoTransform(self.geotransform)  # dem_geotrans)
        dataset.SetProjection(self.wkt_)
        return dataset

    def copy_and_transform(self, zoom=1.0):
        # Probalby doesn't handle angled reference frames correctly.
        copy = self.clone_traits(copy='deep')

        x_center = self.geotransform[0] + self.geotransform[1] * self.x_size * .5
        x_new_spacing = self.geotransform[1] / zoom
        copy.geotransform[0:3] = x_center - x_new_spacing * self.x_size * .5, x_new_spacing, 0

        y_center = self.geotransform[3] + self.geotransform[5] * self.y_size * .5
        y_new_spacing = self.geotransform[5] / zoom
        copy.geotransform[3:6] = y_center - y_new_spacing * self.y_size * .5, 0, y_new_spacing

        return copy

class AbstractDataLayer(HasStrictTraits):

    #===========================================================================
    # Coordinate system maintained by parent simulation
    #===========================================================================
    grid_coordinates = Instance('GridCoordinates')

    #===========================================================================
    # Data Description
    #===========================================================================
    name = Str()  # e.g., 'Temperature'
    units = Str()  # e.g., 'K'
    data_type = Enum([None, 'LHTFL', 'SHTFL', 'GFLUX', 'SSRUN', 'BGRUN', 'TMP',
                 'ALBDO', 'WEASD', 'SNOD', 'SOLMST', 'TSOIL', 'SOLLIQ', 'EVAPTP',
                 'CANOPY', 'WDRUN', 'TMP', 'TMIN', 'TMAX', 'SPFH', 'PRES',
                 'DNSWF', 'DNLWF', 'LAND', 'VEGTYP', 'SOLTYP', 'TERR', 'VGREEN',
                 'RELSOL', 'MINRH', 'APCP', 'EVAPTS', 'NSWRS', 'NLWRS', 'SNOHF',
                 'SNOEV', 'DSWRF', 'DLWRF', 'ASNOW', 'ARAIN', 'EVP', 'SNOM',
                 'AVSFT', 'CNWAT', 'MSTAV', 'EVCW', 'TRANS', 'EVBS',
                 'SBSNO', 'PEVPR', 'ACOND', 'SNOWC', 'CCOND', 'RCS', 'RCT',
                  'RCQ', 'RCSOL', 'RSMIN', 'LAI', 'VEG', 'var250', 'var255'
                  ])
    depth = Enum([None, 'sfc', '0-200 cm down', '0-10 cm down', '0-100 cm down',
                  '0-200 cm down', '10-40 cm down',
                  '40-100 cm down', '100-200 cm down', '2 m above gnd',
                  '10 m above gnd'])

    def __repr__(self):
        return "<%s: %s; Data type: %s; Depth: %s >" % (self.__class__, self.name, self.data_type, self.depth)

    #===========================================================================
    # Enumeration
    #===========================================================================
    is_enumerated = Bool(False)
    enumeration_legend = Dict(key_trait=Int, value_trait=Str)  # e.g., {0:'Sea', 1:'Land'}
    enumeration_colors = Dict(key_trait=Int, value_trait=Tuple((1., 1., 1.)))  # e.g., {0:(128,128,128), 1:(1,50,150)}

    ############################################################################
    # Plotting Info (if not enumerated)
    ############################################################################
    scalar_c_lims = List()
    scalar_cm = Instance(matplotlib.colors.Colormap)

    def _scalar_cm_default(self):
        return matplotlib.cm.get_cmap('gray')

    def reproject_to_grid_coordinates(self, grid_coordinates, interp=gdalconst.GRA_NearestNeighbour):
        """ Reprojects data in this layer to match that in the GridCoordinates
        object. """
        source_dataset = self.grid_coordinates._as_gdal_dataset()
        dest_dataset = grid_coordinates._as_gdal_dataset()
        rb = source_dataset.GetRasterBand(1)
        rb.SetNoDataValue(NO_DATA_VALUE)
        rb.WriteArray(np.ma.filled(self.raster_data, NO_DATA_VALUE))

        gdal.ReprojectImage(source_dataset, dest_dataset,
                            source_dataset.GetProjection(),
                            dest_dataset.GetProjection(),
                            interp)
        dest_layer = self.clone_traits()
        dest_layer.grid_coordinates = grid_coordinates
        rb = dest_dataset.GetRasterBand(1)
        dest_layer.raster_data = np.ma.masked_values(rb.ReadAsArray(), NO_DATA_VALUE)
        return dest_layer

    def export_to_geotiff(self, file_name):
        dest_dataset = self.grid_coordinates._as_gdal_dataset(driver='GTiff',
                                                              file_name=file_name)
        rb = dest_dataset.GetRasterBand(1)
        rb.WriteArray(self.raster_data.filled())
        rb.SetNoDataValue(float(self.raster_data.fill_value))
        rb.SetDescription(self.name)
        rb.SetUnitType(self.units)

    def inpaint(self):
        """ Replace masked-out elements in an array using an iterative image inpainting algorithm. """

        import inpaint
        filled = inpaint.replace_nans(np.ma.filled(self.raster_data, np.NAN).astype(np.float32), 3, 0.01, 2)
        self.raster_data = np.ma.masked_invalid(filled)

    def colormap(self):
        from matplotlib import colors
        import collections
        if self.is_enumerated:
            if self.enumeration_colors:
                d = collections.OrderedDict(sorted(self.enumeration_colors.items()))
                cmap = colors.ListedColormap(d.values())  # + [(0., 0., 0.)])
                bounds = np.array(d.keys() + [d.keys()[-1] + 1]) - .5
                norm = colors.BoundaryNorm(bounds, cmap.N)
                return cmap, norm
            else:
                return None, None
        # not enumerated.
        return self.scalar_cm

    def to_rgba(self):
        data = self.raster_data
        if self.is_enumerated:
            if self.enumeration_colors:
                cmap, norm = self.colormap()
                data2 = norm(data)  # np.clip((data - MIN) / (MAX - MIN), 0, 1)
                rgba = (cmap(data2) * 255).astype(int)
                rgba[:, :, 3] = np.logical_not(data.mask).astype(int) * 255
                return rgba
            else:
                raise NotImplementedError()

        # Not enumerated...
        if self.scalar_c_lims:
            MIN, MAX = self.scalar_c_lims
        else:
            MIN, MAX = data.min(), data.max()

        cm = self.colormap()
        data2 = np.clip((data - MIN) / (MAX - MIN), 0, 1)
        rgba = (cm(data2) * 255).astype(int)
        rgba[:, :, 3] = np.logical_not(data.mask).astype(int) * 255

        return rgba


#===============================================================================
# DifferentiateInput/Staging/Results Layers
#===============================================================================
class InputLayerMixin(HasStrictTraits):
    duration_s = Float()

class StagingLayerMixin(HasStrictTraits):
    pass

class ResultsLayerMixin(HasStrictTraits):
    pass

#===============================================================================
# Raster Data
#===============================================================================
class AbstractRasterDataLayer(AbstractDataLayer):
    raster_data = Array()

    def interp_value(self, lat, lon, indexed=False):
        """ Lookup a pixel value in the raster data, performing linear interpolation
        if necessary. Indexed ==> nearest neighbor (*fast*). """
        (px, py) = self.grid_coordinates.projection_to_raster_coords(lat, lon)
        if indexed:
            return self.raster_data[round(py), round(px)]
        else:
#             from scipy.interpolate import interp2d
#             f_interp = interp2d(self.grid_coordinates.x_axis, self.grid_coordinates.y_axis, self.raster_data, bounds_error=True)
#             return f_interp(lon, lat)[0]
            from scipy.ndimage import map_coordinates
            ret = map_coordinates(self.raster_data, [[py], [px]], order=1)  # linear interp
            return ret[0]

class InputRasterDataLayer(InputLayerMixin, AbstractRasterDataLayer):
    pass

class StagingRasterDataLayer(StagingLayerMixin, AbstractRasterDataLayer):
    pass

class ResultsRasterDataLayer(ResultsLayerMixin, AbstractRasterDataLayer):
    pass

#===============================================================================
# Point Data
#===============================================================================
class PointMeasurement(HasStrictTraits):
    pass  # Uncertain how to define coord system and how this measurement looks...

class AbstractPointDataLayer(AbstractDataLayer):
    point_measurements = List(Instance(PointMeasurement))

class InputPointDataLayer(InputLayerMixin, AbstractPointDataLayer):
    pass

class StagingPointDataLayer(StagingLayerMixin, AbstractPointDataLayer):
    pass

class ResultsPointDataLayer(ResultsLayerMixin, AbstractPointDataLayer):
    pass


