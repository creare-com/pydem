import os
from pydem import process_manager
import shutil
import rasterio
import numpy as np

'''
Run this script to confirm that pyDEM runs identically in the docker container.
To do run:
- volume mount the pyDEM repository to the docker container
- navigate to /pydem/test
- run script by using this command `python test_docker_container.py`
- the script will process the test file, `test_NN032_033_elev.tif`, in the /test directory 
- it will create a `test_docker_container_results` directory
- it will then compare the newly processed data to old processed data
- finally it will remove the `test_docker_container_results` directory
'''

def run_pydem_with_setting_dict():
    os.makedirs('test_docker_container_results')
    output_path = 'test_docker_container_results'

    pm = process_manager.ProcessManager(n_workers=100, in_path='.')
    pm.process_twi()
    # save results
    # pm.save_non_overlap_data(output_path)
    pm.save_non_overlap_data_geotiff('float32', new_path=output_path, overview_type='average')
    results_path = 'results.zarr'
    shutil.rmtree(results_path)
    print("Done")
    return output_path

def compare_raster_files(file1_path, file2_path):
    """
    Compares two raster files to see if they are the same.

    Parameters:
    file1_path (str): The path to the first raster file.
    file2_path (str): The path to the second raster file.

    Returns:
    bool: True if the files are the same, False otherwise.
    """
    with rasterio.open(file1_path) as src1, rasterio.open(file2_path) as src2:
        
        # Compare data
        for band in range(1, src1.count + 1):
            data1 = src1.read(band)
            data2 = src2.read(band)
            if not np.array_equal(data1, data2):
                print(f"Data does not match in band {band} for files:\n  {file1_path}\n  {file2_path}")
                return False
    print(f"The raster files are identical:\n  {file1_path}\n  {file2_path}")

if __name__ == '__main__':
    results_path = run_pydem_with_setting_dict()
    compare_raster_files(os.path.join(results_path,"elev.tiff"), "testtiff/test_elev.tiff" )
    compare_raster_files(os.path.join(results_path,"aspect.tiff"), "testtiff/test_aspect.tiff" )
    compare_raster_files(os.path.join(results_path,"slope.tiff"), "testtiff/test_slope.tiff" )
    compare_raster_files(os.path.join(results_path,"uca.tiff"), "testtiff/test_uca.tiff" )
    compare_raster_files(os.path.join(results_path,"twi.tiff"), "testtiff/test_twi.tiff" )
    shutil.rmtree(results_path)