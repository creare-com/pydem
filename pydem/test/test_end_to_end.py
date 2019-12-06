import numpy as np
from pydem.dem_processing import DEMProcessor

class TestEndtoEnd(object):
    elev = np.array([[0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1],
                     [2, 2, 3, 2, 2],
                     [3, 3, 4, 3, 3],
                     [3, 4, 4, 4, 3]
                     ])
    mag = np.array([[-1.        , -1.        ,  5.        , -1.        , -1.        ],
        [ 5.        ,  5.        ,  5.        ,  5.        ,  5.        ],
        [ 5.        ,  5.        , 10.        ,  5.        ,  5.        ],
        [ 5.        ,  5.        ,  7.07106781,  5.        ,  5.        ],
        [ 4.95      ,  5.        ,  3.53553391,  5.        ,  3.50017857]])
    ang = np.array([[-1.        , -1.        ,  1.57079633, -1.        , -1.        ],
        [ 1.57079633,  1.57079633,  1.57079633,  1.57079633,  1.57079633],
        [ 1.57079633,  1.57079633,  1.57079633,  1.57079633,  1.57079633],
        [ 1.57079633,  1.57079633,  0.78539816,  1.57079633,  1.57079633],
        [ 1.57079633,  1.57079633,  0.78539816,  1.57079633,  2.35619449]])

    uca = np.array([[0.16, 0.16, 0.12, 0.2 , 0.08],
       [0.12, 0.12, 0.08, 0.16, 0.04],
       [0.08, 0.08, 0.04, 0.12, 0.04],
       [0.04, 0.04, 0.04, 0.04, 0.04],
       [0.04, 0.04, 0.04, 0.04, 0.04]])

    def test_simple_slope(self):
        dp = DEMProcessor(elev=self.elev)
        mag, ang = dp.calc_slopes_directions()
        np.testing.assert_array_almost_equal(mag, self.mag)
        np.testing.assert_array_almost_equal(ang, self.ang)

        uca = dp.calc_uca()
        np.testing.assert_array_almost_equal(uca, self.uca)

if __name__ == '__main__':
    te2e = TestEndtoEnd()
    te2e.test_simple_slope()
