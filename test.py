from pydem import compute_manager
#from multiprocessing import freeze_support

#freeze_support()
pm = compute_manager.ProcessManager(in_path='test_mf/chunks', n_workers=1, _debug=False)
pm.process_twi()
pm.save_non_overlap_data()
pm.process_overviews(out_path=pm.out_path_noverlap)

print ("done")