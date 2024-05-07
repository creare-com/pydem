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

import glob
import os
import numpy as np
import shutil
import tempfile

from pydem import process_manager, utils_test_pydem, DEMProcessor, utils
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
        cls.DIR_NAME_MF = os.path.join(tempfile.mkdtemp(), 'test_multifile')
        cls.DIR_NAME_SF = cls.DIR_NAME_MF.replace("_multifile", "_singlefile")
        try:
            shutil.rmtree(cls.DIR_NAME_MF)
            shutil.rmtree(cls.DIR_NAME_SF)
        except FileNotFoundError:
            pass
        utils_test_pydem.mk_test_multifile(
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
        dp_single.dX2[:] = 1
        dp_single.dY2[:] = 1
        dp_single.calc_twi()
        cls.dp_single = dp_single

        process_manager.DEBUG = True

    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree(cls.DIR_NAME_SF)
            shutil.rmtree(cls.DIR_NAME_MF)
        except FileNotFoundError:
            pass

    def setup_case(self, nx_grid, ny_grid, overlap):
        utils_test_pydem.mk_test_multifile(
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
        pm = process_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()

        np.testing.assert_array_almost_equal(self.dp_single.uca[1:-1, 1:-1], pm.out_file_noverlap['uca'][1:-1, 1:-1])
        self.teardown_case()

    def test_5x4_2overlap(self):
        self.setup_case(5, 4, 2)
        pm = process_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()

        np.testing.assert_array_almost_equal(self.dp_single.uca[1:-1, 1:-1], pm.out_file_noverlap['uca'][1:-1, 1:-1])
        self.teardown_case()

    def test_5x4_3overlap(self):
        self.setup_case(5, 4, 3)
        pm = process_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()

        np.testing.assert_array_almost_equal(self.dp_single.uca[1:-1, 1:-1], pm.out_file_noverlap['uca'][1:-1, 1:-1])
        self.teardown_case()

    def test_3x3_1overlap(self):
        self.setup_case(3, 3, 1)
        pm = process_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()

        np.testing.assert_array_almost_equal(self.dp_single.uca[1:-1, 1:-1], pm.out_file_noverlap['uca'][1:-1, 1:-1])
        self.teardown_case()

    def test_3x4_1overlap(self):
        self.setup_case(3, 4, 1)
        pm = process_manager.ProcessManager(
            in_path=os.path.join(self.DIR_NAME_MF, 'chunks'),
            n_workers=self.N_WORKERS,
            _debug=False,
        )
        pm.process_twi()
        pm.save_non_overlap_data()

        np.testing.assert_array_almost_equal(self.dp_single.uca[1:-1, 1:-1], pm.out_file_noverlap['uca'][1:-1, 1:-1])
        self.teardown_case()


class TestEndtoEndCardinal(object):
    elev = np.array([
        [1, 1, 1, 1, 1,],
        [2, 2, 2, 2, 2,],
        [3, 3, 3, 3, 3,],
        [4, 4, 4, 4, 4,],
        [5, 5, 5, 5, 5,]])
    # Note, the reason there are two -1's in the last row is because of flats handling.
    # The corner flows out, so we cannot determine its direction --> -1.
    # The edges of flats have direction (because the edge has a downstream point) so we need
    # To expand the edges by a pixel to find flats... as a consequence this corner pixel is grabbed
    # and its neighbor is assigned the wrong value.
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
        slc = (slice(0, None), slice(0, None))
        dp = DEMProcessor(elev=self.elev[slc], fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc])
        np.testing.assert_array_almost_equal(ang, self.ang[slc])
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc])

        # Reverse
        slc = (slice(None, None, -1), slice(0, None))
        dp = DEMProcessor(elev=self.elev[slc], fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc])
        #np.testing.assert_array_almost_equal(ang, self.ang[slc] + np.pi)
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc])

        # Transpose
        slc = (slice(0, None), slice(0, None))
        dp = DEMProcessor(elev=self.elev[slc].T, fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc].T)
        #np.testing.assert_array_almost_equal(ang, self.ang[slc] + np.pi)
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc].T)

        # Transpose-reverse
        slc = (slice(None, None, -1), slice(None, None, -1))
        dp = DEMProcessor(elev=self.elev[slc].T, fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc].T)
        #np.testing.assert_array_almost_equal(ang, self.ang[slc] + np.pi)
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc].T)

class TestEndtoEndDiagonal(object):
    elev = np.array([
        [1, 2, 3, 4, 5,],
        [2, 3, 4, 5, 6,],
        [3, 4, 5, 6, 7,],
        [4, 5, 6, 7, 8,],
        [5, 6, 7, 8, 9,]])
    # Note, the reason there are two -1's in the last row is because of flats handling.
    # The corner flows out, so we cannot determine its direction --> -1.
    # The edges of flats have direction (because the edge has a downstream point) so we need
    # To expand the edges by a pixel to find flats... as a consequence this corner pixel is grabbed
    # and its neighbor is assigned the wrong value.
    ang = np.array([
        [3/4, 3/4, 3/4, 3/4, 1],
        [3/4, 3/4, 3/4, 3/4, 3/4],
        [3/4, 3/4, 3/4, 3/4, 3/4],
        [3/4, 3/4, 3/4, 3/4, 3/4],
        [1/2, 3/4, 3/4, 3/4, 3/4]]) * np.pi
    r2 = np.sqrt(2)
    mag = np.array([
        [r2, r2, r2, r2, 1],
        [r2, r2, r2, r2, r2],
        [r2, r2, r2, r2, r2],
        [r2, r2, r2, r2, r2],
        [1, r2, r2, r2, r2]])

    uca = np.array([
        [5, 4, 3, 3, 1],
        [4, 4, 3, 2, 1],
        [3, 3, 3, 2, 1],
        [3, 2, 2, 2, 1],
        [1, 1, 1, 1, 1]])

    def test_simple_slope(self):
        slc = (slice(0, None), slice(0, None))
        dp = DEMProcessor(elev=self.elev[slc], fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc])
        np.testing.assert_array_almost_equal(ang, self.ang[slc])
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc])

        # Reverse
        slc = (slice(None, None, -1), slice(0, None))
        dp = DEMProcessor(elev=self.elev[slc], fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc])
        #np.testing.assert_array_almost_equal(ang, self.ang[slc] + np.pi)
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc])

        # Transpose
        slc = (slice(0, None), slice(0, None))
        dp = DEMProcessor(elev=self.elev[slc].T, fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc].T)
        #np.testing.assert_array_almost_equal(ang, self.ang[slc] + np.pi)
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc].T)

        # Transpose-reverse
        slc = (slice(None, None, -1), slice(None, None, -1))
        dp = DEMProcessor(elev=self.elev[slc].T, fill_flats=False)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag[slc].T)
        #np.testing.assert_array_almost_equal(ang, self.ang[slc] + np.pi)
        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca[slc].T)


if __name__ == '__main__':
    te2e = TestEndtoEndDiagonal()
    te2e.test_simple_slope()

    tmf = TestMultiFileEndToEnd()
    tmf.setup_class()
    tmf.test_3x3_2overlap()
    print('done')
