import numpy as np
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
from KalmanFilter import KalmanFilter


class TestKF(unittest.TestCase):

    def test_kf_construction(self):
        x = 0.2
        v = 0.3
        kalman_filter = KalmanFilter(init_x = x, init_v = v, init_acc_var=1.2)
        self.assertAlmostEqual(kalman_filter._x[0], x)
        self.assertAlmostEqual(kalman_filter._x[1], v)
        self.assertAlmostEqual(kalman_filter.pos, x)
        self.assertAlmostEqual(kalman_filter.velo, v)

    def test_prediction(self):
        x = 0.2
        v = 0.3
        kalman_filter = KalmanFilter(init_x = x, init_v = v, init_acc_var=1.2)
        kalman_filter.prediction(dt=0.1)
        self.assertEqual(kalman_filter.cov.shape, (2,2))
        self.assertEqual(kalman_filter.mean.shape, (2,))

    def test_increasing_uncertainty_after_prediction(self):
        x = 0.2
        v = 0.3
        kalman_filter = KalmanFilter(init_x = x, init_v = v, init_acc_var=1.2)

        for i in range(10):
            det_before = np.linalg.det(kalman_filter._P)
            kalman_filter.prediction(dt=0.1)
            det_after = np.linalg.det(kalman_filter._P)
            self.assertGreater(det_after, det_before)

    def test_update_function_construction(self):
        x = 0.2
        v = 0.3
        kalman_filter = KalmanFilter(init_x = x, init_v = v, init_acc_var=1.2)
        kalman_filter.update(meas_val=0.1, meas_var=0.1)
    

    def test_decrease_uncertainty_after_prediction(self):
        x = 0.2
        v = 0.3
        kalman_filter = KalmanFilter(init_x = x, init_v = v, init_acc_var=1.2)

        det_before = np.linalg.det(kalman_filter.cov)
        kalman_filter.update(meas_val=0.1, meas_var=0.1)
        det_after = np.linalg.det(kalman_filter.cov)

        self.assertLess(det_after, det_before)
