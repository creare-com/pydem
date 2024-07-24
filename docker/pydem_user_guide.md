# pyDEM Terrain Data Processing User Guide

This document provides instructions on how to set up a Docker container with pyDEM and process your own terrain data to create a Digital Elevation Model. By following this guide, you should end up with a directory containing a `results.zarr` directory and 5 GeoTIFF files: `twi.tiff`, `uca.tiff`, `slope.tiff`, `aspect.tiff`, and `elev.tiff`.

## Setup

1. **Download the pyDEM Repository**
   - Download `Dockerfile` file, which defines the Docker image, to `folder-location/`, where "folder-location" is a folder name of your choice.

2. **Build the Docker Image**
   - To set up the pyDEM container, build the Docker image using the Dockerfile. Change directories to `folder-location/` (from above). Run the following command in the command line:
     ```sh
     docker build -t pydem_app .
     ```

3. **Verify Docker Image Creation**
   - A Docker image named `pydem_app` should now be available. It will have pyDEM installed as well as `ipython`.

## Creating a Container

1. **Create a Docker Container**
   -  You can create a Docker container from the image using the following command, but it will not be useful unless you mount your data in the next step.
     ```sh
     docker run -it --rm pydem_app:latest bash
     ```
   - This command spins up an interactive Docker container that you can access from the command line. Exit the container by typing `exit`. **Warning:** Exiting the container will remove it, but you can create a new one anytime.

2. **Mount Your Terrain Data**
   - Before creating your Docker container, mount your terrain data from your filesystem. Modify the Docker run command to include a volume mount that links the directory containing your data to a directory in the container. The working directory of the container is `/app`, so creating a subdirectory here works great. Use the following command:
     ```sh
     docker run -it --rm -v /path/to/your/data:/app/data pydem_app:latest bash
     ```
   - Your data directory should now be live-mounted into the container. Any changes made to your data directory will be reflected both in your file system and in the container.
   - Check file permissions if you encounter issues with accessing your data or writing data.

## Running pyDEM

1. **Run pyDEM with Default Settings**
   - Once your container is running and your terrain data is mounted, create an output directory such as `/app/output_directory`. Use `mkdir /app/output_directory` to create this new directory.
    - **Warning:** Once you run pyDEM, a `results.zarr` directory will be created in your output directory regardless of pyDEM successfully or unsuccesfully processing the your data. Remove `results.zarr` between runs. If pyDEM sees a `results.zarr` directory in your input directory, it will use the data in this directory and create a duplicate even if you change the data files. To avoid this, choose an output directory that is not in the directory with your source data like `/app/output_directory`.
   - You can begin processing the data with the following example code:
     ```python
     from pydem import process_manager

     path = '/app/data'
     output_path = '/app/output_directory'

     pm = process_manager.ProcessManager(n_workers=10, in_path=path)
     pm.process_twi()
     pm.save_non_overlap_data(output_path)
     pm.save_non_overlap_data_geotiff('float32', new_path=output_path, overview_type='average')
     ```
   - This code processes all GeoTIFFs in the specified directory and returns the files: `twi.tiff`, `uca.tiff`, `slope.tiff`, `aspect.tiff`, `elev.tiff`, and `results.zarr`.

2. **Adjusting pyDEM Settings**
   - Change the `n_workers` argument to adjust the resources pyDEM uses and to alter the runtime.
   - The default settings for pyDEM are set for 30m terrain data. For higher resolution data, you may need to change these settings to fill flats and drain pits (these lead to nan values).

3. **Customizing DEM Processor Settings**
   - The DEM processor is the class in the pyDEM package that controls how the Digital Elevation Model (DEM) data is calculated.
   - The default settings of the DEM processor are set for 30m resolution elevation data and should work for resolutions higher than 30m
   - If your data has a higher resolution than 30m, you should consider changing the following settings:
     - `drain_pits_max_iter`
     - `drain_pits_max_dist`
   - Increasing these values will cause longer runtimes but better TWI results with fewer nan values.
   - It is recommended to scale the `drain_pits_max_dist` by `30 m / your data's meter resolution`
     - Continue to tweak this parameter if you notice several pits (nans) are not draining.
   - Once you have your `drain_pits_max_dist` set, you will need to increase the `drain_pits_max_iter` so the DEM processor iterates enough times for the water to drain across each pixel.
     - Increase the `drain_pits_max_iter` until your TWI results appear steady and no longer changes as your increase this parameter's value
   - Refer to the pyDEM `README.md` for all DEM processor settings and descriptions of each.

4. **Overwrite Default Settings**
   - Here is some example code of how to change the DEM Processor settings.
   - It is recommended to use the container's installed ipython application to run this code.
   - Create a dictionary with the settings you want to change:
     ```python
     DEM_processor_settings = {
         "drain_pits_max_iter": 1000,
         "drain_pits_max_dist": 100
     }
     ```
   - Use the following example code to apply these new settings:
     ```python
     from pydem import process_manager

     path = '/app/data'
     output_path = '/app/output_directory'
     DEM_processor_settings = {
         "drain_pits_max_iter": 1000,
         "drain_pits_max_dist": 100
     }

     pm = process_manager.ProcessManager(n_workers=10, in_path=path)
     pm.dem_proc_kwargs = DEM_processor_settings
     pm.process_twi()
     pm.save_non_overlap_data(output_path)
     pm.save_non_overlap_data_geotiff('float32', new_path=output_path, overview_type='average')
     ```

This guide should help you set up the pyDEM Docker container and process your terrain data efficiently. If you have any questions or run into issues, refer to the pyDEM README.md for more detailed information on available settings and usage.
