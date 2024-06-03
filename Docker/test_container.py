import os
from pydem import process_manager
import shutil

def make_dir(path, directory_name):
    # Construct the full path for the directory and subdirectory
    full_path = os.path.join(path, directory_name)
    
    # Check if the directory exists, if not create it along with any necessary parents
    os.makedirs(full_path, exist_ok=True)

def run_pydem_with_setting_dict(path, settings_dict=None):

    for i, setting in enumerate(settings_dict):
        pm = process_manager.ProcessManager(n_workers=100, in_path=path)
        keys = list(setting.keys())

        if settings_dict != None:
            print(setting)
            dp_kwargs = setting
            # dem_setting = keys[0]+ '_' + str(setting[keys[0]]) #+ '_' + keys[1]+ '_' + str(setting[keys[1]])
            dem_setting = '_'.join(f"{key}_{setting[key]}" for key in setting)
            make_dir(path, dem_setting)
            output_path = os.path.join(path, dem_setting)
            pm.dem_proc_kwargs = dp_kwargs
        else:
            make_dir(path, 'default')
            output_path = os.path.join(path, 'default')

        pm.process_twi()
        # save results
        pm.save_non_overlap_data(output_path)
        pm.save_non_overlap_data_geotiff('float32', new_path=output_path, overview_type='average')
        results_path = os.path.join(path,'results.zarr')
        shutil.rmtree(results_path)
        print("Done")


path = 'data'
settings_dict = [
    { 
     "drain_pits_max_iter":500, 
     "drain_pits_max_dist":64
     }
]
run_pydem_with_setting_dict(path, settings_dict)

# docker run -it --rm -v /path/to/your/data:/app/data pydem_app:latest bash
