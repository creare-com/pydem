from pydem import compute_manager
import copy
##from multiprocessing import freeze_support

##freeze_support()
pm = compute_manager.ProcessManager(in_path='test_mf/chunks', n_workers=1, _debug=False)
#pm.process_twi()
#pm.save_non_overlap_data()
#pm.process_overviews(out_path=pm.out_path_noverlap)

#print ("done")

from pydem import DEMProcessor
from pydem.utils import dem_processor_from_raster_kwargs
import numpy as np

kwargs = dem_processor_from_raster_kwargs(pm.elev_source_files[0])

dp = DEMProcessor(**kwargs)
dp_part = DEMProcessor(**kwargs)

uca_init = np.zeros_like(dp.elev)
uca_edge_init = [
    {'left': np.ones(dp.elev.shape[0]), 'right': np.zeros(dp.elev.shape[0]),
     'top': np.zeros(dp.elev.shape[1]), 'bottom': np.zeros(dp.elev.shape[1])},  #data
    {'left': np.ones(dp.elev.shape[0], bool), 'right': np.zeros(dp.elev.shape[0], bool), 
     'top': np.zeros(dp.elev.shape[1], bool), 'bottom': np.zeros(dp.elev.shape[1], bool)}, # done
    {'left': np.ones(dp.elev.shape[0], bool), 'right': np.zeros(dp.elev.shape[0], bool),
     'top': np.zeros(dp.elev.shape[1], bool), 'bottom': np.zeros(dp.elev.shape[1], bool)}  # todo
    ]

dp.calc_uca(edge_init_data=uca_edge_init, uca_init=uca_init)

uca_edge_init_1 = copy.deepcopy(uca_edge_init)
uca_edge_init_2 = copy.deepcopy(uca_edge_init)
uca_edge_init_1[1]['left'][:dp.elev.shape[0] // 2] = 0 
uca_edge_init_2[1]['left'][dp.elev.shape[0] // 2:] = 0
dp_part.calc_uca(edge_init_data=uca_edge_init_1, uca_init=uca_init,
                 plotflag=True)
uca1 = dp_part.uca.copy()
done1 = dp_part.edge_done.copy()
todo1 = dp_part.edge_todo.copy()
matshow(np.abs(uca1[1:-1, 1:-1])); colorbar(); title('part 1')
matshow((done1)); title('part 1 done')
matshow((todo1)); title('part 1 todo')
dp_part.calc_uca(edge_init_data=uca_edge_init_2, uca_init=uca_init)
uca2 = dp_part.uca.copy()
done2 = dp_part.edge_done.copy()
todo2 = dp_part.edge_todo.copy()
matshow(np.log10(uca2[1:-1, 1:-1])); colorbar(); title('part 2')
matshow((done2)); title('part 2 done')
matshow((todo2)); title('part 2 todo')


matshow(np.abs((uca1 + uca2)[1:-1, 1:-1])); colorbar(); title('part')
matshow(np.abs(dp.uca[1:-1, 1:-1])); colorbar(); title('full')
matshow((uca1+uca2)[1:-1, 1:-1] - dp.uca[1:-1, 1:-1], cmap='bwr'); colorbar(); title('part - full')

