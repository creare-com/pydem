# pyDEM: A python digital elevation model analysis package
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
number of independent pyDEM processes. Note that pyDEM depends on TauDEM for
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
The `ProcessManager` class orchestrates multiple processes to compute TWI over multiple tiles in parallel on the same machine. Example usage is as follows:
```python
  from pydem.processing_manager import ProcessManager
  elevation_source_path = r'/home/twi-users/elevation'
  manager = ProcessManager(
    n_workers=64, # Number of worker processes to use
    in_path=elevation_source_path,
    out_path='/home/twi-users/temporary_compute_storage/'  # Where to save intermediate data
    dem_proc_kwargs={},  # dictionary of values used to initialize DemProcessor
  )
  # Start the processing
  manager.process_twi()  # If this fails (e.g. machine goes down), can restart to pick up where it left off
```

You can also call individual parts of the processing:
```python
manager.compute_grid()  # Figures out how elevation in folder are tiled
manager.process_elevation()  # Fills flats, handles pits, fixes artifacts in elevation data
manager.process_aspect_slope()
manager.process_uca() # Computes upstream contributing area (UCA) for individual tiles (embarassingly parallel)
manager.process_uca_edges() # Fixes upstream flow contribution accross tile edges -- iteratively
manager.process_twi()  # Call all the above functions internally, but skips any work already done
```

Finally, to export results to a single GeoTiff with overviews, use:
```python
manager.save_non_overlap_data_geotiff(
  'float32',  # Numpy recognized filetype
  new_path=output_path,
  keys=['elev', 'uca', 'aspect', 'slope', 'twi'], # list can contain a subset of these
  overview_type='average')
```

> Note: The name `non_overlap_data` comes from an implementation quirk. The temporary data is saved as
    a zarr file. This zarr file OVERLAPS data at the edges of tiles to make it easier to compute UCA
    across edges


#### 2.1.3 DEMProcessor options

The following options are used by the DEMProcess object. They can be modified by setting the value before processing.

    dem_proc = DEMProcessor(filename_to_elevation_geotiff)
    dem_proc.fill_flats = False


*Slopes & Directions*

 * `chunk_size_slp_dir`: Chunk size for slopes_directions calculation. Default `512`.
 * `chunk_overlap_slp_dir`: Overlap to use for resolving slopes and directions at chunk edges. Default `4`.
 * `fill_flats`: Fill/interpolate the elevation for flat regions before calculating slopes and directions. The direction cannot be calculated in regions where the slope is 0 because the nominal elevation is all the same. This can happen in very gradual terrain or in lake and river beds, particularly when the input elevation is composed of integers. When `True`, the elevation is interpolated in those regions so that a reasonable slope and direction can be calculated. Default `True`.
 * `fill_flats_below_sea`: Interpolate the elevation for flat regions that are below sea level. Water will never flow out of these "pits", so in many cases you can ignore these regions and achieve faster processing times. Default `False`.
 * `fill_flats_source_tol`: When filling flats, the algorithm finds adjacent "source" pixels and "drain" pixels for each flat region and interpolates the elevation using these data points. This sets the tolerance for the elevation of source pixels above the flat region (i.e. shallow sources are used as sources but not steep cliffs). Default `1`.
 * `fill_flats_peaks`: Interpolate the elevation for flat regions that are "peaks" (local maxima). These regions have a higher elevation than all adjacent pixels, so there are no "source" pixels to use for interpolation. When `True`, a single pixel is selected approximately in the center of the flat region as the "peak"/"source". Default `True`.
 * `fill_flats_pits`: Interpolate the elevation for flat regions that are "pits" (local minima). These regions have a lower elevation than all adjacent pixels, so there are no "drain" pixels to use for interpolation. When `True`, a single pixel is selected approximately in the center of the flat region as the "pit"/"drain". Default `True`.

 *UCA*

 * `resolve_edges`: Ensure edge UCA is continuous across chunks. Default `True`.
 * `chunk_size_uca`: Chunk size for uca calculation. Default `512`.
 * `chunk_overlap_uca`: Overlap to use for resolving uca at chunk edges. Default `32`.
 * `drain_pits`: Drain from "pits" to nearby but non-adjacent pixels. Pits have no lower adjacent pixels to drain to directly. *Note that with `fill_flats_pits` off, this setting will still drain each pixel in large flat regions, but it may be slower and produces less reasonable results.* Default `True`.
 * `drain_pits_max_iter`: Maximum number of iterations to look for drain pixels for pits. Generally, "nearby drains" for a pit/flat region are found by expanding the region upward/outward iteratively. Default `100`.
 * `drain_pits_max_dist`: Maximum distance in coordnate-space to (non-adjacent) drains for pits. Pits that are too far from another pixel with a lower elevation will not drain. Default `20`.
 * `drain_pits_max_dist_XY`: Maximum distance in real-space to (non-adjacent) drains for pits. Pits that are too far from another pixel with a lower elevation will not drain. This filter is applied after `drain_pits_max_dist`; if the X and Y resolution are similar, this filter is generally unnecessary. Default `None`.
 * `drain_flats`: *[Deprecated, replaced by `drain_pits`]* Drains flat regions and pits by draining all pixels in the region to an arbitrary pixel in the region and then draining that pixel to the border of the flat region. Ignored if `drain_pits` is `True`. Default `False`.
 * `apply_uca_limit_edges`: Mark edges as completed if the maximum UCA is reached when resolving drainage across edges. Default `False`. If True, it may speed up large calculations.
 * `uca_saturaion_limit`: Default `32`.

 *TWI*

 * `apply_twi_limits`: When calculating TWI, limit TWI to max value. Default `False`.
 * `apply_twi_limits_on_uca`: When calculating TWI, limit UCA to max value. Default `False`.

 *Other*

  * `save_projection`: Default `EPSG:4326`.

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
* `process_manager.py`: Implements a class that manages the calculation of TWI for a directory of files.
  * Manages the calculation of the upstream contributing area that drains across tile edges.
  * Stores errors in the processing.
  * Allows multiple processes to work on the same directory without causing conflicts.
* `test_pydem.py`: A few helper utilities that create analytic test-cases used to develop/test pyDEM.
* `utils.py`: A few helper utility functions.
  * Renames files in a directory (deprecated, no longer needed).
  * Parses file names.
  * Wraps some `gdal` functions for reading and writing geotiff files.
  * Sorts the rows in an array.
* `cyfuncs`: Directory containing cythonized versions of python functions in `dem_processing.py`. These should be compiled during installation.
  * `cyfuncs.cyutils.pyx`: Computationally efficient implementations of algorithms used to calculate upstream contributing area.
* `examples`: Directory containing a few examples, along with an end-to-end test of the cross-tile calculations.
  * `examples.compare_tile_to_chunk.py`: Compares the calculation of the upstream contributing area over a full tile compared to multiple chunks in a file. This tests that the upstream contributing area calculation correctly drains across tile edges.
    * `examples.process_manager_directory.py`: This shows how to use the `ProcessingManager` to calculate all of the elevation files within a directory.
* `aws`: Directory containing experiment for running PyDEM on Amazon Web Services
* `pydem.test.test_end_to_end.py`: A few integration tests. Can be run from the commandline using the `pytest` package: `cd pydem; cd test; pytest .`

## 4. References
Tarboton, D. G. (1997). A new method for the determination of flow directions and upslope areas in grid digital elevation models. Water resources research, 33(2), 309-319.

Ueckermann, Mattheus P., et al. (2015). "pyDEM: Global Digital Elevation Model Analysis." In K. Huff & J. Bergstra (Eds.), Scipy 2015: 14th Python in Science Conference. Paper presented at Austin, Texas, 6 - 12 July (pp. 117 - 124). [http://conference.scipy.org/proceedings/scipy2015/mattheus_ueckermann.html](http://conference.scipy.org/proceedings/scipy2015/mattheus_ueckermann.html)

## 5. Attributions
pyDEM uses [lib.pyx](https://github.com/alexlib/openpiv-python/blob/cython_safe_installation/openpiv/c_src/lib.pyx) from the [OpenPIV](https://github.com/alexlib/openpiv-python) project for [inpainting](pydem/pydem/reader/inpaint.pyx) missing values in the final outputs.
