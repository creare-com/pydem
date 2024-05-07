# -*- coding: utf-8 -*-
"""
   Copyright 2015-2024 Creare

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
    from pydem.utils_test_pydem import get_test_data
    from pydem.dem_processing import DEMProcessor
    from pydem.utils_test_pydem import make_file_names

    import sys, os
    if len(sys.argv) <= 1:
        testnum = 13  # Default test number

    NN = 64  # Resolution of tile

    # Get test data and create DEMProcessor
    elev, ang, test_data = get_test_data(testnum, NN)

    dem_proc1 = DEMProcessor(elev=elev, fill_flats=False)
    twi1 = dem_proc1.calc_twi()
    dem_proc1b = DEMProcessor(elev=elev, fill_flats=True)
    twi1b = dem_proc1b.calc_twi()
    # Calculate TWI

    # Stretch the latitude direction
    dem_proc2 = DEMProcessor(elev=elev, dY=2, dX=1)
    twi2 = dem_proc2.calc_twi()

    # plot results
    import matplotlib.pyplot as plt
    plt.matshow(twi1); plt.colorbar()
    plt.title('TWI calculated from Array, dx/dy determined from array shape')
    plt.matshow(twi1b); plt.colorbar()
    plt.title('Elevation from Array (after fixing flats)')
    plt.matshow(twi2); plt.colorbar()
    plt.title('TWI calculated from Array, dx/dy determined from input lat/lon')

    plt.show()
