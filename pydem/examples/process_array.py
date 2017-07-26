# -*- coding: utf-8 -*-
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
if __name__ == "__main__":
    from pydem.test_pydem import get_test_data
    from pydem.dem_processing import DEMProcessor
    from pydem.test_pydem import make_file_names

    import sys, os
    if len(sys.argv) <= 1:
        testnum = 13  # Default test number

    NN = 64  # Resolution of tile

    # Get test data and create DEMProcessor
    elev, ang, test_data = get_test_data(testnum, NN)
    elev = elev.raster_data
    
    dem_proc1 = DEMProcessor(elev)
    # Calculate TWI
    twi1 = dem_proc1.calc_twi()
    
    # Stretch the latitude direction
    lats = [0, 2]  # degrees
    lons = [0, 1]  # degrees
    dem_proc2 = DEMProcessor((elev, lats, lons))
    twi2 = dem_proc2.calc_twi()

    # plot results
    import matplotlib.pyplot as plt
    plt.matshow(twi1); plt.colorbar()
    plt.title('TWI calculated from Array, dx/dy determined from array shape')
    plt.matshow(dem_proc1.data); plt.colorbar()
    plt.title('Elevation from Array (after fixing flats)')
    plt.matshow(twi2); plt.colorbar()
    plt.title('TWI calculated from Array, dx/dy determined from input lat/lon')
    
    plt.show()
    
    if not os.path.exists('twi'):
        os.mkdir('twi')
    dem_proc1.save_twi('.')
    dem_proc2.save_twi('.')
    print 'done'