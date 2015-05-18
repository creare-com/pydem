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


    import sys
    if len(sys.argv) <= 1:
        testnum = 30  # Default test number

    NN = 64  # Resolution of tile
    
    # Get test data and create DEMProcessor
    elev, ang, test_data = get_test_data(testnum, NN)
    filename = make_file_names(testnum, NN)['fel']
    dem_proc = DEMProcessor(filename, dx_dy_from_file=False)
    
    # Calculate magnitude and direction of slopes
    mag, direction = dem_proc.calc_slopes_directions()

    # Calculate UCA/TWI on entire tile
    area1 = dem_proc.calc_uca(0)
    twi1 = dem_proc.calc_twi()

    # Calculate UCA/TWI on chunks
    dem_proc.resolve_edges = True  # If false, won't do edge resolution
    dem_proc.chunk_size_uca = 9  # Chunk size
    dem_proc.chunk_overlap_uca = 3  #Overlap between chunks
    area2 = dem_proc.calc_uca(0)  
    twi2 = dem_proc.calc_twi()

    # Calculate UCA/TWI on chunk and apply UCA limit on edges
    dem_proc.apply_uca_limit_edges = True
    area3 = dem_proc.calc_uca(0)
    twi3 = dem_proc.calc_twi()
    
    # plot results
    import matplotlib.pyplot as plt
    plt.matshow(area1); plt.colorbar()
    plt.title('UCA Calculated on Full Tile')
    plt.matshow(area2 - area1); plt.colorbar()
    plt.title('UCA Calculation: (on chunk) - (on full tile)')
    plt.matshow(area3 - area2); plt.colorbar()
    plt.title('UCA Calculation: (on chunk, limited) - (on chunk)')
    plt.show()
    
    
    plt.show()
    