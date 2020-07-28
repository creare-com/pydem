import os
import glob
from matplotlib.pylab import *
from pydem import compute_manager, test_pydem, DEMProcessor, utils
from pydem.compute_manager import EDGE_SLICES
import shutil
from multiprocessing import freeze_support

if 1: #__name__ == '__main__':
    #try:
        #os.chdir(os.path.dirname(__file__))
        #DIR_NAME = os.path.join(os.path.dirname(__file__), 'test_multifile')
    #except:
    DIR_NAME = os.path.join('C:\\temp', 'test_multifile')

    DIR_NAME2 = DIR_NAME.replace("_multifile", "_singlefile")
    
    try:
        shutil.rmtree(DIR_NAME)
        shutil.rmtree(DIR_NAME2)
    except FileNotFoundError:
        pass
    
    TEST_NUM = 33
    OVERLAP = 2
    NN = 32
    N_WORKERS = 1
    DEBUG = True
    
    
    test_pydem.mk_test_multifile(
        TEST_NUM,
        NN,
        DIR_NAME,
        nx_overlap=OVERLAP,
        ny_overlap=OVERLAP,
        nx_grid=3,
        ny_grid=3
    )
    test_pydem.mk_test_multifile(
        TEST_NUM,
        NN,
        DIR_NAME2,
        nx_overlap=0,
        ny_overlap=0,
        nx_grid=1,
        ny_grid=1
    )
    
    # Compute single tile
    fn = glob.glob(os.path.join(DIR_NAME2, 'chunks', '*'))[0]
    dp_single = DEMProcessor(fn)
    dp_single.dX[:] = 1
    dp_single.dY[:] = 1
    #dp_single.calc_uca(plotflag=True)
    twi = dp_single.calc_twi()
    #matshow(dp_single.uca); colorbar()
    #title('single')   
    #show(True)

    # Compute the mf manually -- this is not generalized -- very specific to this particular case
    #fns = glob.glob(os.path.join(DIR_NAME, 'chunks', '*.tif'))
    #fns.sort()
    #dps = [DEMProcessor(fn) for fn in fns]
    #for dp in dps:
        #dp.dX[:] = 1
        #dp.dY[:] = 1
        ##dp.calc_twi()

    #dps = np.array(dps, object).reshape(3, 3)[::-1, ::-1]
    #dp = dps[1, 1]
    #uca_true = dp_single.uca[11:23, 11:23]     
    #dp = dps[1, 2]
    #uca_true = dp_single.uca[11:23, 23:] 
    #dp.uca_true = uca_true
    #dp.calc_uca(plotflag=True)
    #uca = dp.fix_self_edge_pixels(None, None, None, dp.uca)
    #uca = dp.fix_edge_pixels(None, None, None, uca, initialize=False)
    
    
    # Compute MF
    
    
    #freeze_support()
    pm = compute_manager.ProcessManager(
        in_path=os.path.join(DIR_NAME, 'chunks'),
        n_workers=N_WORKERS,
        _debug=DEBUG,
        _true_uca=dp_single.uca
    )
    pm.compute_grid()
    pm.compute_grid_overlaps()    
    pm.process_twi()
    
    #stop
    pm.save_non_overlap_data()
    elev = pm.out_file_noverlap['elev'][:]
    elevov = pm.out_file['elev'][:]
    #pm.process_overviews(out_path=pm.out_path_noverlap)
    
    matshow(dp_single.twi - pm.out_file_noverlap['twi']); colorbar()
    
    matshow(dp_single.uca - pm.out_file_noverlap['uca']); colorbar()
    title('single - multi')
    matshow(pm.out_file_noverlap['uca']); colorbar()
    title('multi')
    matshow(dp_single.uca); colorbar()
    title('single')
    show()
    
    def get_cell(ind, attr='twi'):
        return getattr(dp_single, attr)[pm.grid_slice_noverlap[pm.grid_id2i[ind[0], ind[1]]]]
    
    # Compute the mf manually -- this is not generalized -- very specific to this particular case
    fns = glob.glob(os.path.join(DIR_NAME, 'chunks', '*.tif'))
    fns.sort()
    dps = [DEMProcessor(fn) for fn in fns]
    for dp in dps:
        dp.dX[:] = 1
        dp.dY[:] = 1
        dp.calc_twi()

    dps = np.array(dps, object).reshape(3, 3)[::-1, ::-1]
    dp = dps[1, 1]
    dp.calc_uca(plotflag=True)
    uca_true = dp_single.uca[11:23, 11:23]
    
    
    dps = np.array(dps, object).reshape(4, 3)[::-1, ::-1]
    
    uca_truet = dp_single.uca[15:25, 10:23]
    dpt = dps[2, 1]
    
    dp = dps[3, 1]
    uca_init = dp.uca.copy()
    
    uca_true = dp_single.uca[23:32, 10:23]
    uca_file = pm._true_uca_overlap    
    keys = ['left', 'right', 'top', 'bottom']
    edge_slice = pm.edge_data[pm.grid_id2i[3, 1]]
    out_slice = pm.grid_slice[pm.grid_id2i[3, 1]]
    
    edge_init_data = {key: uca_file[edge_slice[key]] for key in keys}
    edge_init_done = {key: (uca_file[edge_slice[key]]*0 + (key == 'top')).astype(bool) for key in keys}
    edge_init_todo = {key: (uca_file[out_slice][EDGE_SLICES[key]]*0 + (key == 'top')).astype(bool) for key in keys}
        
    new_uca = dp.calc_uca(uca_init=uca_init, edge_init_data=[edge_init_data, edge_init_done, edge_init_todo]) 
    
    # Drain from 1,1, to neighbors
    
    
    
    print ("done")

#from pydem import DEMProcessor
#from pydem import dem_processing
##print(dem_processing.CYTHON)
##dem_processing.CYTHON = False
##print(dem_processing.CYTHON)
#from pydem.utils import dem_processor_from_raster_kwargs
#import numpy as np

#kwargs = dem_processor_from_raster_kwargs(pm.elev_source_files[0])

#dp = DEMProcessor(**kwargs)
#dp_part = DEMProcessor(**kwargs)

#uca_init = np.ones_like(dp.elev)
#uca_edge_init = [
    #{'left': np.ones(dp.elev.shape[0]), 'right': np.zeros(dp.elev.shape[0]),
     #'top': np.zeros(dp.elev.shape[1]), 'bottom': np.zeros(dp.elev.shape[1])},  #data
    #{'left': np.ones(dp.elev.shape[0], bool), 'right': np.zeros(dp.elev.shape[0], bool), 
     #'top': np.ones(dp.elev.shape[1], bool), 'bottom': np.zeros(dp.elev.shape[1], bool)}, # done
    #{'left': np.ones(dp.elev.shape[0], bool), 'right': np.zeros(dp.elev.shape[0], bool),
     #'top': np.zeros(dp.elev.shape[1], bool), 'bottom': np.zeros(dp.elev.shape[1], bool)}  # todo
    #]

#dp.calc_uca(edge_init_data=uca_edge_init, uca_init=uca_init)

#uca_edge_init_1 = copy.deepcopy(uca_edge_init)
#uca_edge_init_2 = copy.deepcopy(uca_edge_init)
#uca_edge_init_1[1]['left'][:dp.elev.shape[0] // 2] = 0 
#uca_edge_init_2[2]['left'][dp.elev.shape[0] // 2:] = 0
#dp_part.calc_uca(edge_init_data=uca_edge_init_1, uca_init=uca_init,
                 #plotflag=True)
#uca1 = dp_part.uca.copy()
#done1 = dp_part.edge_done.copy()
#todo1 = dp_part.edge_todo.copy()
#matshow(np.abs(uca1[1:-1, 1:-1])); colorbar(); title('part 1')
#matshow((done1)); title('part 1 done')
#matshow((todo1)); title('part 1 todo')
#dp_part.calc_uca(edge_init_data=uca_edge_init_2, uca_init=uca_init)
#uca2 = dp_part.uca.copy()
#done2 = dp_part.edge_done.copy()
#todo2 = dp_part.edge_todo.copy()
#matshow(np.log10(uca2[1:-1, 1:-1])); colorbar(); title('part 2')
#matshow((done2)); title('part 2 done')
#matshow((todo2)); title('part 2 todo')


#matshow(np.abs((uca1 + uca2)[1:-1, 1:-1])); colorbar(); title('part')
#matshow(np.abs(dp.uca[1:-1, 1:-1])); colorbar(); title('full')
#matshow((uca1+uca2)[1:-1, 1:-1] - dp.uca[1:-1, 1:-1], cmap='bwr'); colorbar(); title('part - full')

#show()