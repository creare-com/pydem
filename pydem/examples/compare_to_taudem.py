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
import argparse

if __name__ == "__main__":
    import os
    import numpy as np

    from pydem.test_pydem import get_test_data
    from pydem.dem_processing import DEMProcessor
    from pydem.test_pydem import make_file_names
    from pydem.reader.gdal_reader import GdalReader
    from pydem.taudem import taudem

    # Set up argument parsing for command-line usage of this script
    parser = argparse.ArgumentParser(
        description='Compare pydem output to taudem output')
    parser.add_argument('testnum', help='the number of the test elevation to '
                        'process', type=int, default=30, nargs='?')
    parser.add_argument('--variable-coords', '--vc',
                        help='if set, will create a non-'
                        'uniform grid on the elevation', action="store_true")
    args = parser.parse_args()
    testnum = args.testnum
    dx_dy_from_file = args.variable_coords
    testnum += dx_dy_from_file


    NN = 64  # The resolution of the grid (N x N)

    if testnum >= 28:
        filename = make_file_names(testnum, NN)['elev']
    else:
        filename = make_file_names(testnum, NN)['fel']

    elev, ang, test_data = get_test_data(testnum, NN)

    # Make the filenames of the results
    filename2 = make_file_names(testnum, NN)['uca']
    if os.path.exists(filename2):
        uca_file = GdalReader(file_name=filename2)
        uca, = uca_file.raster_layers

    # Make the pydem Processor object and process the elevation
    dem_proc = DEMProcessor(filename, dx_dy_from_file=dx_dy_from_file)
    mag, direction = dem_proc.calc_slopes_directions()
    area = dem_proc.calc_uca(0)

    twi = dem_proc.calc_twi()
#
    # Now process using tauDEM
    os.chdir('testtiff')
    try:
        os.remove('test_sca.tif')
        os.remove('test_ang.tif')
        os.remove('test_slp.tif')
    except:
        pass
    cmd = ('dinfflowdir -fel "%s" -ang "%s" -slp "%s"' % (os.path.split(filename)[-1], 'test_ang.tif', 'test_slp.tif'))
    taudem._run(cmd)
    cmd = ('areadinf -nc -ang "%s" -sca "%s"' % ('test_ang.tif','test_sca.tif'))
    taudem._run(cmd)
    td_sca = GdalReader(file_name='test_sca.tif')
    sca, = td_sca.raster_layers
    td_ang = GdalReader(file_name='test_ang.tif')
    ang, = td_ang.raster_layers
    del td_sca
    del td_ang
    os.chdir('..')


    #=========================================================================
    #     Make the plots
    #=========================================================================

    # Change some of the default matplotlib plot settings
    from matplotlib import rcParams
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12

    from matplotlib import pyplot as plt

    elev2, ang2, test_data2 = get_test_data(0, NN)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.cm import terrain as terrain_cm

    #%%
    n = 4
    subset=[slice(n, -n), slice(n, -n)]
    if dx_dy_from_file:
        X = np.concatenate(([0], np.cumsum(dem_proc.dX)))
        Y = np.concatenate(([0], np.cumsum(dem_proc.dY)))
        maxxy = max(X.max(), Y.max())
        X = X / 1000
        Y = Y / 1000
    else:
        X = np.linspace(0, 1, test_data.raster_data.shape[1])
        Y = np.linspace(0, 1, test_data.raster_data.shape[0])
    X, Y  = np.meshgrid(X, Y)
    Z = (test_data.raster_data + (test_data.raster_data - test_data2.raster_data) * 0)
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.plot_surface(X[subset], Y[subset], Z[subset], cmap=terrain_cm, linewidth=0)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min()]).max() / 2.0
    ax.set_xlim(X.mean() - max_range, X.mean() + max_range)
    ax.set_ylim(Y.mean() - max_range, Y.mean() + max_range)
    plt.xlabel('Longitude distance (km)')
    plt.ylabel('Latitude distance (km)')
    plt.draw()
    plt.title("Elevation")
#    plt.savefig("imgs/%d_Elevation_level.png" % dx_dy_from_file, bbox_inches='tight', pad_inches=0.15, dpi=300 )


#%%
    if dx_dy_from_file:
        X = test_data.grid_coordinates.y_axis
        Y = test_data.grid_coordinates.x_axis
        X, Y  = np.meshgrid(X, Y)

    rcParams['axes.labelsize'] = 24
    rcParams['xtick.labelsize'] = 24
    rcParams['ytick.labelsize'] = 24
    rcParams['legend.fontsize'] = 24

    extents = [Y[subset].min(), Y[subset].max(), X[subset].min(), X[subset].max()]
    clim=[0, 0.30]
    plt.figure()
    if not dx_dy_from_file:
        plt.imshow((area * (area.shape[0] - 1)), cmap='YlGnBu', extent=extents)
    else:
        plt.imshow(area, cmap='YlGnBu', extent=extents)
    plt.title('pyDEM upstream area', fontsize=26)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    if not dx_dy_from_file:
        plt.clim(clim)
        plt.colorbar(ticks=np.linspace(clim[0], clim[1], 5))
    else:
        plt.colorbar(ticks=np.linspace(0, area.max(), 5))

    if not os.path.exists('imgs'):
        os.mkdir('imgs')

    plt.savefig("imgs/%d_pydem_uca.png" % dx_dy_from_file, bbox_inches='tight', pad_inches=0.1, dpi=300 )

    plt.figure()
    plt.imshow(sca.raster_data, cmap='YlGnBu', extent=extents)

    plt.title('tauDEM upstream area', fontsize=26)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.clim(clim)
    plt.colorbar(ticks=np.linspace(clim[0], clim[1], 5))
    plt.savefig("imgs/%d_taudem_uca.png" % dx_dy_from_file, bbox_inches='tight', pad_inches=0.1, dpi=300 )

    # Difference
    plt.figure()
    if not dx_dy_from_file:
        clim = [0, 1e-7]
        plt.imshow(np.abs(sca.raster_data - (area * (area.shape[0] - 1))), cmap='YlGnBu', extent=extents)
    else:
        clim = [0, 0.3]
        plt.imshow(np.abs(sca.raster_data / sca.raster_data.max() - (area / area.max())), cmap='YlGnBu', extent=extents)
    plt.title('Scaled difference', fontsize=26)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.clim(clim)
    if not dx_dy_from_file:
        plt.colorbar(ticks=np.linspace(clim[0], clim[1], 5))
    else:
        plt.colorbar(ticks=np.linspace(clim[0], clim[1], 5))
    plt.savefig("imgs/%d_diff_uca.png" % dx_dy_from_file, bbox_inches='tight', pad_inches=0.1, dpi=300 )
