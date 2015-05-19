# pydem: A python digital elevation model analysis package
-----------------------------------------------------------

PyDEM is a package for topographic (terrain) analysis written in Python (with a
bit of Cython). It takes in digital elevation model (DEM) rasters, and it 
outputs quantities like slope, aspect, upstream area, and topographic wetness
index. PyDEM closely follows the approach taken by TauDEM to calculate these
quantities. It is designed to be fast, easily extensible, and capable of
handling very large datasets.

PyDEM can be used both from the command-line, or programmatically via its Python
API. Examples of both usages are given below and in the examples directory. It
can operate on individual elevation rasters, or on entire directories
simultaneously. Most processing steps can be performed in parallel with any
number of independent PyDEM processes. Note that PyDEM depends on TauDEM for
certain steps (e.g., pitfilling) and it also makes extensive use of the GDAL
library for working with geospatial rasters.

## 1. Installation
For installation notes, see [install.md](INSTALL.md)

## 2. Basic Usage
Examples and tests can be found in the `pydem\examples` directory.

### 2.1 Python Module Usage
#### 2.1.1. Calculate quantities on a single elevation tile
Import the `DEMProcessor` class:

    from pydem.dem_processing import DEMProcessor

Set the path to a pit filled elevation file in WGS84 coordinates:

    filename_to_elevation_geotiff = 'test.tiff'

Instantiate an instance of the `DEMProcessor` class:

    dem_proc = DEMProcessor(filename_to_elevation_geotiff)

The following three commands **do** ***not*** **need to be called in order**.

Calculate the aspect and slope magnitude:

    mag, aspect = dem_proc.calc_slopes_directions()

Calculate the upstream contributing area:

    uca = dem_proc.calc_uca()

Calculate the TWI:

    twi = dem_proc.calc_twi()

#### 2.1.2 Calculate TWI on a directory of elevation tiles
Import the `ProcessManager` class:

    from pydem.processing_manager import ProcessManager

Set the path to the directory of pit-filled WGS84 geotiff elevation files. ***Important***: The files in this directory needs to follow the naming convention used. See `examples.process_manager_directory.py` for an example of using `utils.rename_files` to rename elevation files.

    elevation_source_path = r'C:\test_directory'

Set the target location (where files will be saved):

    # This is the default location if unspecified. 
    elevation_target_path = r'C:\test_directory\processed_data'

Instantiate an instance of the `ProcessManager` class:

    pm = ProcessManager(elevation_source_path, elevation_target_path)

Process the elevation files in that directory:

    pm.process()

Topographic wetness index calculated for each elevation file in the supplied directory
will be located in `C:\test_directory\processed_data\twi`. These TWI files will not have
edge effects on edges interior to the data set. 

#### 2.1.3 Calculate a custom quantity on a directory of elevation tiles
Import and Instantiate a `ProcessManager`:
    from pydem.processing_manager import ProcessManager
    # Will save results to default path: 'C:\test_directory\processed_data'
    pm = ProcessManager(r'C:\test_directory')
    
Define a custom command. For example, to calculate hill-shading using gdal:

    def command(source_elevation_file, target_file_save_location):
        cmd = ['gdaldem', 'hillshade', '-s', '111120',
               '-compute_edges', '-co',  'BIGTIFF=YES', '-of',
               'GTiff', '-co', 'compress=lzw', '-co', 'TILED=YES',
               esfile, fn]
        print '<'*8, ' '.join(cmd), '>'*8
        status = subprocess.call(cmd)
        return status

Process directory of files using custom command:
    pm.process_command(command, save_name='hillshade_files')

The results of this operation will be found in `C:\test_directory\processed_data\hillshade_files`.

### 2.2 Commandline Usage
When installing pydem using the provided setup.py file, the commandline utilities `TWIDinf`, `AreaDinf`, and `DinfFlowDir` are registered with the operating system. 

#### TWIDinf : 

    usage: TWIDinf-script.py [-h] [--save-all]
                         Input_Pit_Filled_Elevation [Input_Number_of_Chunks]
                         [Output_D_Infinity_TWI]
    
    Calculates a grid of topographic wetness index which is the log_e(uca / mag),
    that is, the natural log of the ratio of contributing area per unit contour
    length and the magnitude of the slope. Note, this function takes the elevation
    as an input, and it calculates the slope, direction, and contributing area as
    intermediate steps.
    
    positional arguments:
      Input_Pit_Filled_Elevation
                        The input pit-filled elevation file in geotiff format.
      Input_Number_of_Chunks
                        The approximate number of chunks that the input file
                        will be divided into for processing (potentially on
                        multiple processors).
      Output_D_Infinity_TWI
                        Output filename for the topographic wetness index.
                        Default value = twi.tif .
    
    optional arguments:
      -h, --help            show this help message and exit
      --save-all, --sa      If set, will save all intermediate files as well.

#### AreaDinf : 

    usage: AreaDinf-script.py [-h] [--save-all]
                          Input_Pit_Filled_Elevation [Input_Number_of_Chunks]
                          [Output_D_Infinity_Specific_Catchment_Area]
    
    Calculates a grid of specific catchment area which is the contributing area
    per unit contour length using the multiple flow direction D-infinity approach.
    Note, this is different from the equivalent tauDEM function, in that it takes
    the elevation (not the flow direction) as an input, and it calculates the
    slope and direction as intermediate steps.

    positional arguments:
      Input_Pit_Filled_Elevation
                        The input pit-filled elevation file in geotiff format.
      Input_Number_of_Chunks
                        The approximate number of chunks that the input file
                        will be divided into for processing (potentially on
                        multiple processors).
      Output_D_Infinity_Specific_Catchment_Area
                        Output filename for the flow direction. Default value = uca.tif .
    
    optional arguments:
      -h, --help            show this help message and exit
      --save-all, --sa      If set, will save all intermediate files as well.


#### DinfFlowDir : 

    usage: DinfFlowDir-script.py [-h]
                             Input_Pit_Filled_Elevation
                             [Input_Number_of_Chunks]
                             [Output_D_Infinity_Flow_Direction]
                             [Output_D_Infinity_Slope]
    
    Assigns a flow direction based on the D-infinity flow method using the
    steepest slope of a triangular facet (Tarboton, 1997, "A New Method for the
    Determination of Flow Directions and Contributing Areas in Grid Digital
    Elevation Models," Water Resources Research, 33(2): 309-319).
    
    positional arguments:
      Input_Pit_Filled_Elevation
                        The input pit-filled elevation file in geotiff format.
      Input_Number_of_Chunks
                        The approximate number of chunks that the input file
                        will be divided into for processing (potentially on
                        multiple processors).
      Output_D_Infinity_Flow_Direction
                        Output filename for the flow direction. Default value = ang.tif .
      Output_D_Infinity_Slope
                        Output filename for the flow direction. Default value = mag.tif .
    
    optional arguments:
      -h, --help            show this help message and exit

## 3. Description of package Contents
* `commandline_utils.py` : Contains the functions that wrap the python modules into command line utilities.
* `dem_processing.py`: Contains the main algorithms. 
  * Re-implements the D-infinity method from Tarboton (1997).  
  * Implements a new upstream contributing area algorithm. This performs essentially the same task as previous upstream contributing area algorithms, but with some added functionality. This version deals with areas where the elevation is flat or has no data values and can be updated from the edges without re-calculating the upstream contributing area for the entire tile. 
  * Re-implements the calculation of the [Topographic Wetness Index](http://en.wikipedia.org/wiki/Topographic_Wetness_Index).
* `processing_manager.py`: Implements a class that manages the calculation of TWI for a directory of files.
  * Manages the calculation of the upstream contributing area that drains across tile edges.
  * Stores errors in the processing.
  * Allows multiple processes to work on the same directory without causing conflicts.
* `test_pydem.py`: A few helper utilities that create analytic test-cases used to develop/test pydem.
* `utils.py`: A few helper utility functions.
  * Renames files in a directory.
  * Parses file names.
  * Wraps some `gdal` functions for reading and writing geotiff files.
  * Sorts the rows in an array.
* `cyfuncs`: Directory containing cythonized versions of python functions in `dem_processing.py`. 
  * `cyfuncs.cyutils.pyx`: Computationally efficient implementations of algorithms used to calculate upstream contributing area.
* `examples`: Directory containing a few examples, along with an end-to-end test of the cross-tile calculations.
  * `examples.compare_tile_to_chunk.py`: Compares the calculation of the upstream contributing area over a full tile compared to multiple chunks in a file. This tests that the upstream contributing area calculation correctly drains across tile edges.
  * `examples.compare_to_taudem.py`: This compares the calculation of magnitude and aspect to taudem's algorithms. This validates that the algorithms are correctly implemented from Tarboton (1997), and also shows the differences when taking the change in coordinates into account.
  * `examples.cross-tile_process_manager_test.py`: End-to-end test to make sure that the cross-tile calculations are correctly performed.
  * `examples.process_manager_directory.py`: This shows how to use the `ProcessingManager` to calculate all of the elevation files within a directory. 
* `reader`: Directory containing python code used to deal with opening and closing geotiff files. Essentially wraps `gdal` to provide simpler usage.
  * `reader.gdal_reader.py`: Contains the `GDALReader class used to read and write geotiff files. 
  * `reader.inpaint.pyx`: Cython function used to fill no-data values in geotiffs.
  * `reader.my_types.py`: Defines classes used to deal with different grid coordinate systems.
* `taudem`: Directory containing a copy of taudem for convenience.

## 4. References
Tarboton, D. G. (1997). A new method for the determination of flow directions and upslope areas in grid digital elevation models. Water resources research, 33(2), 309-319.
