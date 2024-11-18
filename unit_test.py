import unittest
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from test import rotate_and_project, bandpass_filter, cross_correlate_top_left

# FILE: test_test.py


class TestCrossCorrelation(unittest.TestCase):
    def setUp(self):
        # Initialize test variables
        self.I_filtered = np.random.rand(100, 100).astype(np.float32)
        self.volume = np.random.rand(100, 100, 100).astype(np.float32)
        self.phi_angles = np.linspace(0, 180, 10)
        self.theta_angles = np.linspace(0, 180, 10)
        self.psi_angles = np.linspace(0, 180, 10)
        self.projection_shape = (100, 100)
        
    def test_cross_correlation(self):
        ncc_max = np.full_like(self.I_filtered, -np.inf, dtype=np.float32)
        ncc_mean = np.zeros_like(self.I_filtered, dtype=np.float32)
        ncc_M2 = np.zeros_like(self.I_filtered, dtype=np.float32)
        n = 0
        
        comp_start = time.time()
        
        def process_psi(psi):
            nonlocal ncc_max, ncc_mean, ncc_M2, n
            for phi, theta in zip(self.phi_angles, self.theta_angles):
                projection = rotate_and_project(self.volume, phi, theta, psi)
                if projection.shape != self.projection_shape:
                    continue
                projection = projection - np.mean(projection)
                proj_filtered = bandpass_filter(projection, low_sigma=1, high_sigma=5)
                T_mean = np.mean(proj_filtered)
                T_std = np.std(proj_filtered)
                T_norm = (proj_filtered - T_mean) / T_std
                cross_corr = cross_correlate_top_left(self.I_filtered, T_norm)
                ncc_max = np.maximum(ncc_max, cross_corr)
                n += 1
                delta = cross_corr - ncc_mean
                ncc_mean += delta / n
                delta2 = cross_corr - ncc_mean
                ncc_M2 += delta * delta2
        
        with ThreadPoolExecutor() as executor:
            executor.map(process_psi, self.psi_angles)
        
        ncc_variance = ncc_M2 / (n - 1) if n > 1 else np.zeros_like(self.I_filtered, dtype=np.float32)
        
        final = time.time()
        elapsed_time = final - comp_start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Cross-correlation computation taken for all {n} templates: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
        
        self.assertTrue(np.all(ncc_max >= -np.inf))
        self.assertTrue(np.all(ncc_mean >= 0))
        self.assertTrue(np.all(ncc_variance >= 0))

if __name__ == '__main__':
    unittest.main()