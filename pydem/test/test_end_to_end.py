import glob
import os
import numpy as np
import shutil

from pydem import compute_manager, test_pydem, DEMProcessor, utils
from pydem.dem_processing import DEMProcessor

class TestMultiFileEndToEnd(object):
    TEST_NUM = 33
    NN = 32
    N_WORKERS = 1
    DIR_NAME_MF = ''
    DIR_NAME_SF = ''
    
    dp_single = None
    
    @classmethod
    def setup_class(cls):
        cls.DIR_NAME_MF = os.path.join('C:\\temp', 'test_multifile')
        cls.DIR_NAME_SF = cls.DIR_NAME_MF.replace("_multifile", "_singlefile")
        try:
            shutil.rmtree(cls.DIR_NAME_MF)
            shutil.rmtree(cls.DIR_NAME_SF)
        except FileNotFoundError:
            pass
        test_pydem.mk_test_multifile(
            cls.TEST_NUM,
            cls.NN,
            cls.DIR_NAME_SF,
            nx_overlap=0,
            ny_overlap=0,
            nx_grid=1,
            ny_grid=1
        )
        fn = glob.glob(os.path.join(cls.DIR_NAME_SF, 'chunks', '*'))[0]
        dp_single = DEMProcessor(fn)
        dp_single.dX[:] = 1
        dp_single.dY[:] = 1        
        dp_single.calc_twi()
        cls.dp_single = dp_single
        
        compute_manager.DEBUG = True
    
    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree(cls.DIR_NAME_SF)
            shutil.rmtree(cls.DIR_NAME_MF)
        except FileNotFoundError:
            pass
    
    def setup_case(self, nx_grid, ny_grid, overlap):
        test_pydem.mk_test_multifile(
            self.TEST_NUM,
            self.NN,
            self.DIR_NAME_MF,
            nx_overlap=overlap,
            ny_overlap=overlap,
            nx_grid=nx_grid,
            ny_grid=ny_grid
        )
        
    def teardown_case(self):
        shutil.rmtree(self.DIR_NAME_MF)
        
    def test_3x3_2overlap(self):
        self.setup_case(3, 3, 2)
        pm = compute_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()
        
        np.testing.assert_array_almost_equal(self.dp_single.uca, pm.out_file_noverlap['uca'])
        self.teardown_case()
        
    def test_5x4_2overlap(self):
        self.setup_case(5, 4, 2)
        pm = compute_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()
        
        np.testing.assert_array_almost_equal(self.dp_single.uca, pm.out_file_noverlap['uca'])
        self.teardown_case()        
        
    def test_5x4_3overlap(self):
        self.setup_case(5, 4, 3)
        pm = compute_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()
        
        np.testing.assert_array_almost_equal(self.dp_single.uca, pm.out_file_noverlap['uca'])
        self.teardown_case()            

class TestEndtoEnd(object):
    elev = np.array([
        [1, 1, 1, 1, 1,], 
        [2, 2, 2, 2, 2,],
        [3, 3, 3, 3, 3,],
        [4, 4, 4, 4, 4,],
        [5, 5, 5, 5, 5,]])
    ang = np.array([
        [-1.        , -1.        ,  1.57079633, -1.        , -1.        ],
        [ 1.57079633,  1.57079633,  1.57079633,  1.57079633,  1.57079633],
        [ 1.57079633,  1.57079633,  1.57079633,  1.57079633,  1.57079633],
        [ 1.57079633,  1.57079633,  1.57079633,  1.57079633,  1.57079633],
        [ 1.57079633,  1.57079633,  1.57079633,  1.57079633,  1.57079633]])
    mag = np.array([
        [-1., -1.,  1., -1., -1.],
        [ 1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.]])

    uca = np.array([
        [np.nan, np.nan, 5, np.nan , np.nan],
        [4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3],
        [2, 2, 2, 2, 2],
        [1, 1, 1, 1, 1]])

    def test_simple_slope(self):
        dp = DEMProcessor(elev=self.elev, fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag)
        np.testing.assert_array_almost_equal(ang, self.ang)

        uca = dp.calc_uca()
        #uca = dp.fix_self_edge_pixels(None, None, None, uca)
        #uca = dp.fix_edge_pixels(None, None, None, uca, initialize=False)
        np.testing.assert_array_almost_equal(uca, self.uca)

if __name__ == '__main__':
    te2e = TestEndtoEnd()
    te2e.test_simple_slope()
