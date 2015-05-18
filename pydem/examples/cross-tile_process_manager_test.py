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
    import numpy as np
    import os

    from pydem.processing_manager import ProcessManager
    from pydem.test_pydem import make_test_files, mk_test_multifile


    #%% Make the test case files
    NN = [300, 400, 660, 740]
    test_num = 32
    testdir = 'testtiff'
    make_test_files(NN, testnum=test_num, testdir=testdir, plotflag=False)
    mk_test_multifile(test_num, NN, testdir, nx_grid=3, ny_grid=4,
                      nx_overlap=16, ny_overlap=32)

    path = r'testtiff\chunks'

    # Remove a couple of these files so that we only have 4 tiles, and we
    # know where they should drain to
    files = os.listdir(path)
    files.sort()
    for i, fil in enumerate(files):
        print i, fil
    delete_ids = [0, 1, 2, 3, 4, 5, 6, 9]
    for d_id in delete_ids:
        os.remove(os.path.join(path, files[d_id]))

    # Create the ProcessManager object
    savepath = r'testtiff\processed_data'
    pm = ProcessManager(path, savepath)
    pm._DEBUG = True  # Save out the magnitude and slope
    pm.elev_source_files.sort()
    esfile = pm.elev_source_files[1]  # Start with lower-left tile and go CCW

    # Start twi calculation for first tile
    fn, status = pm.calculate_twi(esfile,
                              save_path=pm.save_path, do_edges=False)
    edge_init_data, edge_init_done, edge_init_todo = \
        pm.tile_edge.get_edge_init_data(esfile)

    # THe only valuable information here is the edge_init_todo, which is self-set
    # In this case the right edge of the tile is the edge that needs,
    # information, so the right todo should be True
    np.testing.assert_(np.all(edge_init_todo['right'][1:-1])) #don't look at corners
    np.testing.assert_(np.all(~edge_init_todo['left'][1:-1])) #don't look at corners

    # Next we check that the right and top neighbors are correctly set also
    top = pm.tile_edge.neighbors[esfile]['top']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(top)
    np.testing.assert_(np.all(~edge_init_done['bottom'][1:-1])) #don't look at corners
#    stop
    right = pm.tile_edge.neighbors[esfile]['right']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(right)
    np.testing.assert_(np.all(~edge_init_done['left'][1:-1])) #don't look at corners
    topright = pm.tile_edge.neighbors[esfile]['top-right']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(topright)
    np.testing.assert_(np.all(~edge_init_done['left'][1:-1])) #don't look at corners
    np.testing.assert_(np.all(~edge_init_done['bottom'][1:-1])) #don't look at corners
    # pm.tile_edge.visualize_neighbors()

    # do the next tile
    esfile = pm.elev_source_files[0]
    fn, status = pm.calculate_twi(esfile,
                              save_path=pm.save_path, do_edges=False)
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(esfile)
    np.testing.assert_(np.all(~edge_init_todo['right'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['left'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['top'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['bottom'][1:-1]))
    # Next we check that the left and top neighbors are correctly set also
    top = pm.tile_edge.neighbors[esfile]['top']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(top)
    np.testing.assert_(np.all(edge_init_done['bottom']))
    left = pm.tile_edge.neighbors[esfile]['left']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(left)
    np.testing.assert_(np.all(edge_init_done['right']))
    topleft = pm.tile_edge.neighbors[esfile]['top-left']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(topleft)
    np.testing.assert_(np.any(edge_init_done['right']))
    np.testing.assert_(np.any(edge_init_done['bottom']))
    # pm.tile_edge.visualize_neighbors()

    # Do the third tile
    esfile = pm.elev_source_files[2]
    fn, status = pm.calculate_twi(esfile,
                              save_path=pm.save_path, do_edges=False)
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(esfile)
    np.testing.assert_(np.all(~edge_init_todo['right'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['left'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['top'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['bottom'][1:-1]))
    # Next we check that the left and top neighbors are correctly set also
    left = pm.tile_edge.neighbors[esfile]['left']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(left)
    np.testing.assert_(np.all(edge_init_done['right']))
    bottomleft = pm.tile_edge.neighbors[esfile]['bottom-left']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(bottomleft)
    np.testing.assert_(np.any(edge_init_done['right']))
    np.testing.assert_(np.any(edge_init_done['top']))
    # pm.tile_edge.visualize_neighbors()
#    a1 = pm.dem_proc.uca.copy()
#    esfile = pm.elev_source_files[2]
#    coords1 = parse_fn(esfile)
#    imshow(a1, interpolation='none',
#           extent=[coords1[1], coords1[3], coords1[0], coords1[2]]);clim(0, a1.max())
#    crds = pm.tile_edge.edges[left]['right'].get_coordinates()
#    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(left)
#    imshow(edge_init_data['right'][:, None], interpolation='none',
#           extent=[crds[:, 1].min(), crds[:, 1].max()+0.3 / a1.shape[0],
#           crds[:, 0].min(), crds[:, 0].max()]);clim(0, a1.max())
#    xlim(coords1[1], coords1[3])
#    ylim(coords1[0], coords1[2])

    #%%Do the final tile to complete the first round (non-edge resolving)
    esfile = pm.elev_source_files[3]
    fn, status = pm.calculate_twi(esfile,
                              save_path=pm.save_path, do_edges=False)

    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(esfile)
    np.testing.assert_(np.all(~edge_init_todo['right'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['left'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['top'][1:-1]))
    np.testing.assert_(np.any(~edge_init_todo['bottom'][1:-1]))  # mixed on bottom
    np.testing.assert_(np.any(edge_init_todo['bottom'][1:-1]))  # mixed on bottom
    # This one has no neighbors to check (no downstream dependencies)

#    a2 = pm.dem_proc.uca.copy()
#    esfile = pm.elev_source_files[3]
#    coords = parse_fn(esfile)
#    imshow(a2, extent=[coords[1], coords[3], coords[0], coords[2]],
#           interpolation='none');clim(0, a1.max())
#    xlim(coords[1], coords1[3])

    # Now let us start the edge resolution round. There are only 2 tiles that
    # require edge resolution
    # %%
    i = pm.tile_edge.find_best_candidate(pm.elev_source_files)
    np.testing.assert_(i==1) # should be the first tile
    esfile = pm.elev_source_files[i]

    fn, status = pm.calculate_twi(esfile,
                              save_path=pm.save_path, do_edges=True)
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(esfile)
    np.testing.assert_(np.all(~edge_init_todo['right'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['left'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['top'][1:-1]))
    np.testing.assert_(np.all(~edge_init_todo['bottom'][1:-1]))
    # check neihbors
    top = pm.tile_edge.neighbors[esfile]['top']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(top)
    np.testing.assert_(np.all(edge_init_done['bottom'][1:-1])) #don't look at corners
    right = pm.tile_edge.neighbors[esfile]['right']
    edge_init_data, edge_init_done, edge_init_todo = pm.tile_edge.get_edge_init_data(right)
    np.testing.assert_(np.all(edge_init_done['left'][1:-1])) #don't look at corners

    i = pm.tile_edge.find_best_candidate(pm.elev_source_files)
    np.testing.assert_(i==3) # should be the last tile
    esfile = pm.elev_source_files[i]
    fn, status = pm.calculate_twi(esfile,
                              save_path=pm.save_path, do_edges=True)