# pyDEM Terrain Data Processing User Guide
In this document are instructions on how to setup a docker container with pyDEM and process your own terrain data. By following this guide, you should end with a directory that contians a results.zarr directory and 5 geotiff files [`twi.tiff`, `uca.tiff`,`slope.tiff`,`aspect.tiff`,`elev.tiff`]

## Setup 
* Download the pyDEM repository `pydem.zip` and unzip the folder into a directory where your unprocessed terrain data lives

* To setup the pyDEM container, we will need to build the image using the Dockerfile. Build the docker file using this command in the command line: `docker build -t pydem_app .`
  * If you do not have an Ironbank account, the docker build command will fail. Go into the Dockerfile and comment the first line that starts with "FROM". Uncomment this line `FROM python:3.10-slim` and try again.

* A Docker image should now be created called pyDEM_app. It will have pyDEM installed as well as ipython. 

## Creating a Container
* Now that the image is made, we can create a Docker container from the image using this command `docker run -it --rm pydem_app:latest bash`. This command will spin up an interactable Docker container that you can acces in the command line. You can exit the container by typing `exit` into the command line. **Warning:** Exiting the container will remove it, but don't worry a new one can be made.
* Before you create your Docker container, you will need to mount your terrain data from your filesystem  so you have something to process with pyDEM. To do so, you will need to modifity the Docker run command to include a volume mount that will link the directory that contains your data to a directory in the container. The working directory of the container is `/app` so making a subdirectory in here works great. Modify the docker command following this format: `docker run -it --rm -v /path/to/your/data:/app/data pydem_app:latest bash`. Your data directory should now be live mounted into the container. Any changes made to your data directory will be reflected both in your file system and in the container. File permissions sometimes become an issue here so make sure to check those if you are having trouble.

## Running pyDEM
Once you have your container running and your terrain data mounted, you can begin processing the data. 
* To run pyDEM with default settings, use this example code:
```
from pydem import process_manager
path = '/path/to/my/data'
pm = process_manager.ProcessManager(n_workers=10, in_path=path)
pm.process_twi()
pm.save_non_overlap_data(output_path)
pm.save_non_overlap_data_geotiff('float32', new_path=output_path, overview_type='average')
```
This code will process all geotiffs in the directory that you specify and return these files: `twi.tiff`, `uca.tiff`,`slope.tiff`,`aspect.tiff`, `elev.tiff`, and `results.zarr`
* Changing the `n_workers` argument will adjust the resources pyDEM uses as well as change the runtime 
* The default settings for pyDEM were set for 30m terrain data. If your data has a higher resolution you will likely want to change these settings to fill flats and drain pits (these lead to nan values)
* All of the settings for the DEM processor and their descriptions can be found in the `pyDEM README.md`. Most of these settings do not need to be touched. The settings you should tweak are as follows:
  - `drain_pits_max_iter`
  - `drain_pits_max_dist`
* These settings need to be increased for data with more pixels and higher resolution. Increasing the values for these settings will result in longer runtimes, but better results with less nan values. It is recommended to increase these values until the results become steady i.e. the twi does not change as you increase these values. 
* To change the DEM processor settings, you will need to make a dictionary with the setting you want to change as the key and the value of the setting as the entry.
```
DEM_processor_settings = { 
    "drain_pits_max_iter":1000, 
    "drain_pits_max_dist":100
    }
```
* Next, use this example code to overwrite the default settings of the DEM processor using the process manager
```
from pydem import process_manager
path = '/path/to/my/data'
DEM_processor_settings = { 
    "drain_pits_max_iter":1000, 
    "drain_pits_max_dist":100
    }
pm = process_manager.ProcessManager(n_workers=10, in_path=path)
pm.dem_proc_kwargs = DEM_processor_settings
pm.process_twi()
pm.save_non_overlap_data(output_path)
pm.save_non_overlap_data_geotiff('float32', new_path=output_path, overview_type='average')
```
This will give the same outputs as the previous example with the new settings applied.+
