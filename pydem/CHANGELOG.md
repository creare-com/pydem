# 1.0.0 Python 3 release
At long last, `pydem` uses Python 3 instead of Python 2. The new version of `pydem` drops support for Python 2 entirely.

## Enhancements
* `ProcessManager`:
  * Nearly a complete re-implementation
  * No longer depends on a file naming convension to process multiple elevation tiles
  * Now uses the `zarr` file format for intermediate computations
  * A single process oversees multiple workers, making it easy to stop or restart the processing
* `DEMProcessor`:
  * Now uses raw elevation files and conditions them internally without simply filling flats
  * Interpolates flat areas to allow them to drain appropriately
  * Drains pits along a path of minimum elevation
  * Removed in-file chunking capability
* Added basic unit tests (closer to integration tests)
* Migrated from `traits` to `traitlets`
* Migrated from `gdal` to `rasterio`
*