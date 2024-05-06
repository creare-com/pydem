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

def DinfFlowDir():
    import argparse

    parser = argparse.ArgumentParser(
        description='Assigns a flow direction based on the D-infinity flow'
        ' method using the steepest slope of a triangular facet (Tarboton, '
        '1997, "A New Method for the Determination of Flow Directions and '
        'Contributing Areas in Grid Digital Elevation Models," Water Resources'
        ' Research, 33(2): 309-319).')
    parser.add_argument('Input_Pit_Filled_Elevation',
                        help='The input pit-filled elevation file in '
                            'geotiff format.',
                        )
    parser.add_argument('Input_Number_of_Chunks',
                        help="The approximate number of chunks that the input "
                        'file will be divided into for processing (potentially'
                        ' on multiple processors).', type=int, default=1, nargs='?')
    parser.add_argument('Output_D_Infinity_Flow_Direction',
                        help='Output filename for the flow direction. Default'
                        ' value = ang.tif .', default='ang.tif', nargs='?')
    parser.add_argument('Output_D_Infinity_Slope',
                        help='Output filename for the flow direction. Default'
                        ' value = mag.tif .', default='mag.tif', nargs='?')
    args = parser.parse_args()
    fn = args.Input_Pit_Filled_Elevation
    n_chunks = args.Input_Number_of_Chunks
    fn_ang = args.Output_D_Infinity_Flow_Direction
    fn_mag = args.Output_D_Infinity_Slope

    from pydem.dem_processing import DEMProcessor
    dem_proc = DEMProcessor(fn)
    shape = dem_proc.data.shape
    chunk_size = max(shape[0],  shape[1]) / n_chunks
    dem_proc.chunk_size_slp_dir = chunk_size
    dem_proc.calc_slopes_directions()
    dem_proc.save_array(dem_proc.mag, fn_mag, as_int=False)
    dem_proc.save_array(dem_proc.direction, fn_ang, as_int=False)

def AreaDinf():
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculates a grid of specific catchment area which is the'
        ' contributing area per unit contour length using the multiple flow '
        'direction D-infinity approach. Note, this is different from the '
        'equivalent tauDEM function, in that it takes the elevation '
        '(not the flow direction) as an input, and it calculates the '
        'slope and direction as intermediate steps.')
    parser.add_argument('Input_Pit_Filled_Elevation',
                        help='The input pit-filled elevation file in '
                            'geotiff format.',
                        )
    parser.add_argument('Input_Number_of_Chunks',
                        help="The approximate number of chunks that the input "
                        'file will be divided into for processing (potentially'
                        ' on multiple processors).', type=int, default=1,
                        nargs='?')
    parser.add_argument('Output_D_Infinity_Specific_Catchment_Area',
                        help='Output filename for the flow direction. Default'
                        ' value = uca.tif .', default='uca.tif', nargs='?')
    parser.add_argument('--save-all', '--sa',
                        help='If set, will save all intermediate files as well.',
                        action='store_true')
    args = parser.parse_args()
    fn = args.Input_Pit_Filled_Elevation
    n_chunks = args.Input_Number_of_Chunks
    fn_uca = args.Output_D_Infinity_Specific_Catchment_Area
    sa = args.save_all

    from pydem.dem_processing import DEMProcessor
    dem_proc = DEMProcessor(fn)
    shape = dem_proc.data.shape
    chunk_size = max(shape[0],  shape[1]) / n_chunks
    dem_proc.chunk_size_slp_dir = chunk_size
    dem_proc.chunk_size_uca = chunk_size
    dem_proc.calc_slopes_directions()
    if sa:
        dem_proc.save_array(dem_proc.mag, 'mag.tif', as_int=False)
        dem_proc.save_array(dem_proc.direction, 'ang.tif', as_int=False)
    dem_proc.calc_uca()
    dem_proc.save_array(dem_proc.uca, fn_uca, as_int=False)

def TWIDinf():
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculates a grid of topographic wetness index '
        ' which is the log_e(uca / mag), that is, the natural log of the ratio'
        ' of contributing area per unit contour length and the magnitude of '
        ' the slope. Note, this function takes the elevation '
        'as an input, and it calculates the '
        'slope, direction, and contributing area as intermediate steps.')
    parser.add_argument('Input_Pit_Filled_Elevation',
                        help='The input pit-filled elevation file in '
                            'geotiff format.',
                        )
    parser.add_argument('Input_Number_of_Chunks',
                        help="The approximate number of chunks that the input "
                        'file will be divided into for processing (potentially'
                        ' on multiple processors).', type=int, default=1,
                        nargs='?')
    parser.add_argument('Output_D_Infinity_TWI',
                        help='Output filename for the topographic wetness '
                        'index. Default value = twi.tif .', default='twi.tif',
                        nargs='?')
    parser.add_argument('--save-all', '--sa',
                        help='If set, will save all intermediate files as well.',
                        action='store_true')
    args = parser.parse_args()
    fn = args.Input_Pit_Filled_Elevation
    n_chunks = args.Input_Number_of_Chunks
    fn_twi = args.Output_D_Infinity_TWI
    sa = args.save_all

    from pydem.dem_processing import DEMProcessor
    dem_proc = DEMProcessor(fn)
    shape = dem_proc.data.shape
    chunk_size = max(shape[0],  shape[1]) / n_chunks
    dem_proc.chunk_size_slp_dir = chunk_size
    dem_proc.chunk_size_uca = chunk_size
    dem_proc.calc_slopes_directions()
    dem_proc.calc_uca()
    if sa:
        dem_proc.save_array(dem_proc.mag, 'mag.tif', as_int=False)
        dem_proc.save_array(dem_proc.direction, 'ang.tif', as_int=False)
        dem_proc.save_array(dem_proc.uca, 'uca.tif', as_int=False)
    dem_proc.calc_twi()
    dem_proc.save_array(dem_proc.twi, fn_twi, as_int=True)

if __name__ == "__main__":
#    DinfFlowDir()
#    AreaDinf()
    TWIDinf()