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

import gdal
import gdalconst
import os
import numpy as np
import datetime

from traits.api import *
from my_types import GridCoordinates, InputRasterDataLayer, d_wkt_to_name

# I have to define UTC time myself.
ZERO = datetime.timedelta(0)
class UTC(datetime.tzinfo):
    def utcoffset(self, dt): return ZERO
    def tzname(self, dt): return "UTC"
    def dst(self, dt): return ZERO
utc = UTC()

class Reader(HasStrictTraits):
    pass

class DiskReader(Reader):

    file_types = List(Str)  
    
    def update(self):
        """Forces input file to be read."""
        
    def read_file_into_simulation(self, file_name, simulation):
        """Loads data from file directly into a simulation object.
        
        Parameters
        ----------
        file_name: str
            Path to the file to load.
        simulation: dassp.Simulation
            Simulation into which the data should be read.
        
        Raises
        ------
        IOError:
            If file not found.
        RuntimeError:
            If the file cannot be loaded.
        
        .. note:: Overwrites any current input layers in simulation.
        """
        # Check for errors...
        self._read_file_into_simulation(file_name, simulation)


    def _read_file_into_simulation(self, file_name, simulation):
        """Override this with implementation."""
        raise NotImplementedError()

class GdalReader(DiskReader):
    file_types = ListStr([".grib", ".grib2", '.grb', '.gr1', '.tif', '.vrt',
        '.hgt', 'flt', 'adf', '.tiff'])
    acceptable_data_types = ListStr([])
    acceptable_depths = ListStr([])
    file_name = Str()
    _gdal_dataset = Instance(gdal.Dataset)
    grid_coordinates = Property(Array(), depends_on="_gdal_dataset")
    raster_layers = Property(List(Instance(InputRasterDataLayer)),
                      depends_on="_gdal_dataset")

    def summary(self):
        dataset = self._gdal_dataset
        summary_str = "Dataset:\n"
        summary_str += '\tDriver: ' + dataset.GetDriver().ShortName + '/' + \
          dataset.GetDriver().LongName + "\n"
        summary_str += '\tSize is ' + str(dataset.RasterXSize) + 'x' + str(dataset.RasterYSize) + \
              'x' + str(dataset.RasterCount) + "\n"
        summary_str += '\tProjection is ' + str(dataset.GetProjection()) + "\n"
        geotransform = dataset.GetGeoTransform()
        if not geotransform is None:
            summary_str += '\tOrigin = (' + str(geotransform[0]) + ' +' + str(geotransform[3]) + ')'
            summary_str += '\tPixel Size = (' + str(geotransform[1]) + ' +' + str(geotransform[5]) + ')'
        return summary_str

    def update(self):
        """Forces input file to be read."""
        _, _ = self._gdal_dataset, self.grid_coordinates

    def __gdal_dataset_default(self):
        """DiskReader implementation."""

        if not os.path.exists(self.file_name):
            return None
        if os.path.splitext(self.file_name)[1].lower() not in self.file_types:
            raise RuntimeError('Filename %s does not have extension type %s.' % (self.file_name, self.file_types))

        dataset = gdal.OpenShared(self.file_name, gdalconst.GA_ReadOnly)
        if dataset is None:
            raise ValueError('Dataset %s did not load properly.' % self.file_name)

        # Sanity checks.
        assert dataset.RasterCount > 0

        # Seems okay...
        return dataset

    @cached_property
    def _get_grid_coordinates(self):
        dataset = self._gdal_dataset
        geotransform = np.array(dataset.GetGeoTransform())
        assert len(geotransform) == 6
        x_size, y_size = dataset.RasterXSize, dataset.RasterYSize
        assert x_size > 0
        assert y_size > 0

        ds_time = ds_date = None

        wkt_ = dataset.GetProjection()
        wkt = d_wkt_to_name[wkt_]


        return GridCoordinates(x_size=x_size, y_size=y_size,
                               date=ds_date, time=ds_time,
                               geotransform=geotransform,
                               wkt=wkt)

    def _raster_layer_from_raster_band(self, raster_band, arr=None, grid_coordinates=None):
        layer = InputRasterDataLayer()
        if grid_coordinates:
            layer.grid_coordinates = grid_coordinates
        else:
            layer.grid_coordinates = self.grid_coordinates

        # Make sure this dataset doesn't include any features that
        # we're missing...
        # assert raster_band.GetCategoryNames() is None
        assert raster_band.GetColorInterpretation() in [0, 1, 2]  # Not sure what 1 or 2 means.
#        assert raster_band.GetColorTable() is None
#             assert np.count_nonzero(raster_band.GetMaskBand().ReadAsArray()
#                                         < 255) == 0
        assert raster_band.GetRasterCategoryNames() is None
        # assert raster_band.GetOverviewCount() == 0
        # assert raster_band.GetScale() == 1.0
        #assert raster_band.GetUnitType() == ''

        layer.units = ''

        # Read array data.
        if arr is None:
            arr = raster_band.ReadAsArray().copy()
            arr = np.ma.masked_array(arr, np.logical_not(raster_band.GetMaskBand().ReadAsArray()))
            # NAN's:
            # -9999 and 9999 are NaN's... if we eventually get masked data,
            # we'll want to take care of it here.
            if raster_band.GetNoDataValue() is None:
                for nan_value in [-9999, 9999, np.nan]:
                    try:
                        arr = np.ma.masked_equal(arr, nan_value)
                    except:
                        pass
        else:
            arr = np.ma.masked_invalid(arr)
#         if raster_band.GetNoDataValue() is not None:
#             arr = np.ma.masked_equal(arr, raster_band.GetNoDataValue())
        # Check size...
        assert arr.shape[0] == layer.grid_coordinates.y_size
        assert arr.shape[1] == layer.grid_coordinates.x_size

        # Prevent possible rounding errors
        if layer.is_enumerated:
            arr = np.round(arr)

        # It looks valid... assign the raster data to the layer.
        layer.raster_data = arr
        # print "Processed layer:"
        # layer.print_traits()
        return layer

    @cached_property
    def _get_raster_layers(self):
        raster_bands = [self._gdal_dataset.GetRasterBand(i)
                          for i in range(1, self._gdal_dataset.RasterCount + 1)]
        raster_layers = [self._raster_layer_from_raster_band(raster_band)
                         for raster_band in raster_bands]
        return raster_layers

