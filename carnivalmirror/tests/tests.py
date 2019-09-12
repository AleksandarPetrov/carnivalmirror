# Does a number of tests. Should be run from the folder where this file is
# located. Will store diagnostic images in tests

import unittest
import os
import copy
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from carnivalmirror.calibration import Calibration
from carnivalmirror.sampling import ParameterSampler, TriangularParameterSampler, UniformAPPDSampler, ParallelBufferedSampler

HISTOGRAM_TEST_SAMPLES = 2000

def path_here(rel):
    return os.path.join(os.path.dirname(__file__), rel)

class TestCalibrationMethods(unittest.TestCase):

    def setUp(self):
        # Load an image an image and its calibration parameters
        self.test_image = cv2.imread(path_here('test_image_before.png'))
        self.test_image_D = np.array([-0.2967039649743125, 0.06795775093662262, 0.0008927768064001824,
                                      -0.001327854648098482, 0.0])
        self.test_image_K = np.array([336.7755634193813, 0.0, 333.3575643300718, 0.0, 336.02729840829176,
                                      212.77376312080065, 0.0, 0.0, 1.0]).reshape((3, 3))

        # Initialize a calibration object
        self.calibration = Calibration(K=self.test_image_K, D=self.test_image_D,
                                       width=self.test_image.shape[1], height=self.test_image.shape[0])

    def test_print_image_dimensions(self):
        print("Test image dimensions: %s" % str(self.test_image.shape))
        self.assertTrue(True)

    def test_check_ar(self):
        self.assertAlmostEqual(self.calibration.aspect_ratio, 640.0/480)

    def test_scaling_of_K(self):
        self.assertTrue(np.allclose(self.test_image_K, self.calibration.get_K(height=480)))

    def test_storing_of_D(self):
        self.assertTrue(np.allclose(self.test_image_D, self.calibration.get_D()))

    def test_check_undistort_maps_original_size(self):
        width = self.test_image.shape[1]
        height = self.test_image.shape[0]
        undistort_x, undistort_y = self.calibration.undistortion_maps(width=width, height=height)
        self.assertEqual(undistort_x.shape, self.test_image.shape[:2])

    def test_check_undistort_maps_reduced_size(self):
        width = self.test_image.shape[1]
        height = self.test_image.shape[0]
        undistort_x, undistort_y = self.calibration.undistortion_maps(width=width, height=height,
                                                                      map_width=int(width/2), map_height=int(height/2))
        self.assertEqual(undistort_x.shape, (self.test_image.shape[0]/2, self.test_image.shape[1]/2))

    def test_check_undistortion_maps_preserving_original_size(self):
        width = self.test_image.shape[1]
        height = self.test_image.shape[0]
        undistort_x, undistort_y = self.calibration.undistortion_maps_preserving(width=width, height=height)
        self.assertEqual(undistort_x.shape, self.test_image.shape[:2])

    def test_check_undistortion_maps_preserving_reduced_size(self):
        width = self.test_image.shape[1]
        height = self.test_image.shape[0]
        undistort_x, undistort_y = self.calibration.undistortion_maps_preserving(width=width, height=height,
                                                                                 map_width=int(width/2),
                                                                                 map_height=int(height/2))
        self.assertEqual(undistort_x.shape, (self.test_image.shape[0]/2, self.test_image.shape[1]/2))

    def test_rectified_standard_size(self):
        # Apply our rectification
        rect_ours = self.calibration.rectify(self.test_image, mode='standard')
        self.assertEqual(rect_ours.shape, self.test_image.shape)

    def test_rectified_reduced_size(self):
        width = self.test_image.shape[1]
        height = self.test_image.shape[0]

        # Apply our rectification
        rect_ours = self.calibration.rectify(self.test_image, mode='standard',
                                             result_width=int(width/2), result_height=int(height/2))
        self.assertEqual(rect_ours.shape, (self.test_image.shape[0]/2, self.test_image.shape[1]/2, self.test_image.shape[2]))

    def test_rectified_standard_with_ground_truth(self):
        # Apply our rectification
        rect_ours = self.calibration.rectify(self.test_image, mode='standard')

        # Apply directly the OpenCV rectification
        newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(self.test_image_K, self.test_image_D,
                                                                               (self.test_image.shape[1],
                                                                                self.test_image.shape[0]), 1.0)
        map1, map2 = cv2.initUndistortRectifyMap(self.test_image_K, self.test_image_D, np.eye(3), newCameraMatrix,
                                                 (self.test_image.shape[1], self.test_image.shape[0]), cv2.CV_32FC1)
        rect_direct = cv2.remap(self.test_image, map1, map2, cv2.INTER_LANCZOS4)
        cv2.imwrite(path_here("test_rectified_standard_with_ground_truth.png"), np.hstack((rect_direct, rect_ours)))
        self.assertTrue(np.allclose(rect_direct, rect_ours))


    def test_rectified_preserving_with_ground_truth(self):
        # Apply our rectification
        rect_ours = self.calibration.rectify(self.test_image, mode='preserving')

        # Apply directly the OpenCV rectification
        output_resolution = (self.test_image.shape[1], self.test_image.shape[0])
        newCameraMatrix, validPixROI = cv2.getOptimalNewCameraMatrix(self.test_image_K, self.test_image_D,
                                                                               (self.test_image.shape[1],
                                                                                self.test_image.shape[0]), 1.0)
        map1, map2 = cv2.initUndistortRectifyMap(self.test_image_K, self.test_image_D, np.eye(3), newCameraMatrix,
                                                 (self.test_image.shape[1], self.test_image.shape[0]), cv2.CV_32FC1)
        rect_direct = cv2.remap(self.test_image, map1, map2, cv2.INTER_LANCZOS4)

        # Do the preserving on the direct image
        target_aspect_ratio = float(rect_direct.shape[1]) / float(rect_direct.shape[0])
        validROI = {'x': (validPixROI[0], validPixROI[0] + validPixROI[2]),
                    'y': (validPixROI[1], validPixROI[1] + validPixROI[3])}
        valid_aspect_ratio = float(validROI['x'][1] - validROI['x'][0]) / (validROI['y'][1] - validROI['y'][0])

        # If the image is taller than it should, cut its legs and head
        if valid_aspect_ratio < target_aspect_ratio:
            desired_number_of_rows = int(round((validROI['x'][1] - validROI['x'][0]) / target_aspect_ratio))
            cut_top = int(((validROI['y'][1] - validROI['y'][0]) - desired_number_of_rows) / 2)
            cropped_ROI = {
                'x': validROI['x'],
                'y': (validROI['y'][0] + cut_top, validROI['y'][0] + cut_top + desired_number_of_rows)
            }
        else:
            desired_number_of_cols = int(round((validROI['y'][1] - validROI['y'][0]) * target_aspect_ratio))
            cut_left = int(((validROI['x'][1] - validROI['x'][0]) - desired_number_of_cols) / 2)
            cropped_ROI = {
                'x': (validROI['x'][0] + cut_left, validROI['x'][0] + cut_left + desired_number_of_cols),
                'y': validROI['y']
            }

        cropped_image = rect_direct[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
        resized_image = cv2.resize(cropped_image, output_resolution, interpolation=cv2.INTER_LANCZOS4)

        # Calculate the cropped maps (need that for the px movement metric calculation)
        map1 = copy.deepcopy(map1)
        map2 = copy.deepcopy(map2)
        map1 = map1[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
        map1 = cv2.resize(map1, output_resolution, interpolation=cv2.INTER_LANCZOS4)
        map2 = map2[cropped_ROI['y'][0]:cropped_ROI['y'][1], cropped_ROI['x'][0]:cropped_ROI['x'][1]]
        map2 = cv2.resize(map2, output_resolution, interpolation=cv2.INTER_LANCZOS4)

        # Prepare the image output
        rect_ours_gray = cv2.cvtColor(rect_ours, cv2.COLOR_BGR2GRAY)
        resized_direct_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY, dstCn=3)
        overlayed = cv2.addWeighted(rect_ours_gray, 0.5, resized_direct_gray, 0.5, 0)

        cv2.imwrite(path_here("test_rectified_preserving_with_ground_truth.png"),overlayed)
        self.assertTrue(np.allclose(rect_ours_gray, resized_direct_gray, rtol=15, atol=15))

    def test_appd(self):
        perturbed_K = self.test_image_K * 0.98
        perturbed_D = self.test_image_D * 1.01
        perturbed_calibration = Calibration(K=perturbed_K, D=perturbed_D,
                                            width=self.test_image.shape[1], height=self.test_image.shape[0])

        appd1, dm1 = self.calibration.appd(perturbed_calibration,
                                           height=self.test_image.shape[0], width=self.test_image.shape[1],
                                           interpolation=cv2.INTER_LANCZOS4, min_cropped_size=None,
                                           return_diff_map=True)

        appd2, dm2 = perturbed_calibration.appd(self.calibration,
                                           height=self.test_image.shape[0], width=self.test_image.shape[1],
                                           interpolation=cv2.INTER_LANCZOS4, min_cropped_size=None,
                                           return_diff_map=True)


        self.assertTrue(np.isclose(appd1, appd2, rtol=1e-2, atol=1e-2))
        self.assertTrue(np.allclose(dm1, dm2, rtol=1e-2, atol=1e-2))
        self.assertEqual(dm1.shape, (self.test_image.shape[0], self.test_image.shape[1]))

        cv2.imwrite(path_here("test_appd_diff_map.png"),dm1)

    def test_appd_self(self):

        appd, dm = self.calibration.appd(self.calibration,
                                         height=self.test_image.shape[0], width=self.test_image.shape[1],
                                         interpolation=cv2.INTER_LANCZOS4,
                                         min_cropped_size=None, return_diff_map=True)

        self.assertTrue(np.isclose(appd, 0.0, rtol=1e-2, atol=1e-2))
        self.assertTrue(np.allclose(dm, np.zeros_like(dm), rtol=1e-2, atol=1e-2))

    def test_region_of_validity_check(self):
        perturbed_K = self.test_image_K * 2
        perturbed_D = self.test_image_D * 9
        perturbed_calibration = Calibration(K=perturbed_K, D=perturbed_D,
                                            width=self.test_image.shape[1], height=self.test_image.shape[0])

        with self.assertRaises(RuntimeError):
            _, _ = perturbed_calibration.appd(self.calibration,
                                              height=self.test_image.shape[0],
                                              width=self.test_image.shape[1],
                                              interpolation=cv2.INTER_LANCZOS4, min_cropped_size=None,
                                              return_diff_map=True)

    def test_appd_map_resolution_invariance(self):
        perturbed_K = self.test_image_K * 0.98
        perturbed_D = self.test_image_D * 1.02
        perturbed_calibration = Calibration(K=perturbed_K, D=perturbed_D,
                                            width=self.test_image.shape[1], height=self.test_image.shape[0])

        appd_full, dm_full = perturbed_calibration.appd(self.calibration,
                                                        height=self.test_image.shape[0], width=self.test_image.shape[1],
                                                        interpolation=cv2.INTER_LANCZOS4, min_cropped_size=None,
                                                        return_diff_map=True)

        appd_half, dm_half = perturbed_calibration.appd(self.calibration,
                                                        height=int(self.test_image.shape[0]),
                                                        width=int(self.test_image.shape[1]),
                                                        interpolation=cv2.INTER_LANCZOS4, min_cropped_size=None,
                                                        return_diff_map=True, map_width=int(self.test_image.shape[1]/2),
                                                        map_height=int(self.test_image.shape[0]/2))

        self.assertAlmostEqual(appd_full, appd_half, places=3)
        self.assertTrue(np.allclose(dm_full, cv2.resize(dm_half, (dm_full.shape[1], dm_full.shape[0]),
                                                        interpolation=cv2.INTER_LANCZOS4),
                                    rtol = 1e-1, atol = 1e-1))

    def test_normaized_appd_map_resolution_invariance(self):
        perturbed_K = self.test_image_K * 0.98
        perturbed_D = self.test_image_D * 1.02
        perturbed_calibration = Calibration(K=perturbed_K, D=perturbed_D,
                                            width=self.test_image.shape[1], height=self.test_image.shape[0])

        appd_full, dm_full = perturbed_calibration.appd(self.calibration,
                                                        height=self.test_image.shape[0], width=self.test_image.shape[1],
                                                        interpolation=cv2.INTER_LANCZOS4, min_cropped_size=None,
                                                        return_diff_map=True, normalized=True)

        appd_half, dm_half = perturbed_calibration.appd(self.calibration,
                                                        height=int(self.test_image.shape[0]),
                                                        width=int(self.test_image.shape[1]),
                                                        interpolation=cv2.INTER_LANCZOS4, min_cropped_size=None,
                                                        return_diff_map=True, map_width=int(self.test_image.shape[1]/2),
                                                        map_height=int(self.test_image.shape[0]/2), normalized=True)

        self.assertAlmostEqual(appd_full, appd_half, places=3)
        self.assertTrue(np.allclose(dm_full, cv2.resize(dm_half, (dm_full.shape[1], dm_full.shape[0]),
                                                        interpolation=cv2.INTER_LANCZOS4),
                                    rtol = 1e-1, atol = 1e-1))

class TestParameterSampler(unittest.TestCase):

    def setUp(self):
        # Load an image an image and its calibration parameters
        self.test_image = cv2.imread(path_here('test_image_before.png'))
        self.test_image_D = np.array([-0.2967039649743125, 0.06795775093662262, 0.0008927768064001824,
                                      -0.001327854648098482, 0.0])
        self.test_image_K = np.array([336.7755634193813, 0.0, 333.3575643300718, 0.0, 336.02729840829176,
                                      212.77376312080065, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.width = self.test_image.shape[1]
        self.height = self.test_image.shape[0]
        self.ranges = { 'fx': (0.95*336.7755634193813, 1.05*336.7755634193813),
                        'fy': (0.95*336.02729840829176, 1.05*336.02729840829176),
                        'cx': (0.95*333.3575643300718, 1.05*333.3575643300718),
                        'cy': (0.95*212.77376312080065, 1.05*212.77376312080065),
                        'k1': (1.05*-0.2967039649743125, 0.95*-0.2967039649743125),
                        'k2': (0.95*0.06795775093662262, 1.05*0.06795775093662262),
                        'p1': (0.95*0.0008927768064001824, 1.05*0.0008927768064001824),
                        'p2': (1.05*-0.001327854648098482, 0.95*-0.001327854648098482),
                        'k3': (0.0, 0.0)}

        # Initialize a reference calibration object
        self.reference = Calibration(K=self.test_image_K, D=self.test_image_D,
                                     width=self.test_image.shape[1], height=self.test_image.shape[0])

    def test_histogram(self):
        sampler = ParameterSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height)
        t = time.time()
        bins, hist_edges = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES,
                                             n_bins=20, width=self.width, height=self.height,
                                             min_cropped_size=(int(self.width/1.5), int(self.height/1.5)))
        print("ParameterSampler time for %d samples  (full res): %.02f, AVG: %.04f" %
               (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.figure()
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge")
        plt.grid()
        plt.savefig(path_here('ParameterSampler_test_histogram.png'))
        plt.close()

    def test_histogram_normalized(self):
        sampler = ParameterSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height)

        # Full resolution
        t = time.time()
        bins, hist_edges = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES,
                                             n_bins=20, width=self.width, height=self.height, normalized=True,
                                             min_cropped_size=(int(self.width/1.5), int(self.height/1.5)))
        print("ParameterSampler time for %d normalized samples (full res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.figure()
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge", label="full resolution", alpha=0.5)

        # 1/16 resolution
        t = time.time()
        bins, hist_edges, appds = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES, return_values=True,
                                                    n_bins=20, width=int(self.width/4), height=int(self.height/4), normalized=True,
                                                    min_cropped_size=(int(self.width/4/1.5), int(self.height/4/1.5)))
        print("ParameterSampler time for %d normalized samples (1/16 res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge", label="1/16 resolution", alpha=0.5)

        plt.grid()
        plt.legend()
        plt.savefig(path_here('ParameterSampler_test_histogram_normalized.png'))
        plt.close()


        # Save a plot that shows the sample sequence
        plt.figure()
        plt.plot(appds, 'x')
        plt.grid()
        plt.savefig(path_here('ParameterSampler_test_histogram_normalized_sequence.png'))
        plt.close()

class TestTriangularParameterSampler(unittest.TestCase):

    def setUp(self):
        # Load an image an image and its calibration parameters
        self.test_image = cv2.imread(path_here('test_image_before.png'))
        self.test_image_D = np.array([-0.2967039649743125, 0.06795775093662262, 0.0008927768064001824,
                                      -0.001327854648098482, 0.0])
        self.test_image_K = np.array([336.7755634193813, 0.0, 333.3575643300718, 0.0, 336.02729840829176,
                                      212.77376312080065, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.width = self.test_image.shape[1]
        self.height = self.test_image.shape[0]
        self.ranges = { 'fx': (0.95*336.7755634193813, 1.05*336.7755634193813),
                        'fy': (0.95*336.02729840829176, 1.05*336.02729840829176),
                        'cx': (0.95*333.3575643300718, 1.05*333.3575643300718),
                        'cy': (0.95*212.77376312080065, 1.05*212.77376312080065),
                        'k1': (1.05*-0.2967039649743125, 0.95*-0.2967039649743125),
                        'k2': (0.95*0.06795775093662262, 1.05*0.06795775093662262),
                        'p1': (0.95*0.0008927768064001824, 1.05*0.0008927768064001824),
                        'p2': (1.05*-0.001327854648098482, 0.95*-0.001327854648098482),
                        'k3': (0.0, 0.0)}

        # Initialize a reference calibration object
        self.reference = Calibration(K=self.test_image_K, D=self.test_image_D,
                                     width=self.test_image.shape[1], height=self.test_image.shape[0])

    def test_histogram(self):
        sampler = TriangularParameterSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height, reference=self.reference)
        t = time.time()
        bins, hist_edges = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES,
                                             n_bins=20, width=self.width, height=self.height,
                                             min_cropped_size=(int(self.width/1.5), int(self.height/1.5)))
        print("TriangularParameterSampler time for %d samples  (full res): %.02f, AVG: %.04f" %
               (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.figure()
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge")
        plt.grid()
        plt.savefig(path_here('TriangularParameterSampler_test_histogram.png'))
        plt.close()

    def test_histogram_normalized(self):
        sampler = TriangularParameterSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height, reference=self.reference)

        # Full resolution
        t = time.time()
        bins, hist_edges = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES,
                                             n_bins=20, width=self.width, height=self.height, normalized=True,
                                             min_cropped_size=(int(self.width/1.5), int(self.height/1.5)))
        print("TriangularParameterSampler time for %d normalized samples (full res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.figure()
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge", label="full resolution", alpha=0.5)

        # 1/16 resolution
        t = time.time()
        bins, hist_edges, appds = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES, return_values=True,
                                                    n_bins=20, width=int(self.width/4), height=int(self.height/4), normalized=True,
                                                    min_cropped_size=(int(self.width/4/1.5), int(self.height/4/1.5)))
        print("TriangularParameterSampler time for %d normalized samples (1/16 res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge", label="1/16 resolution", alpha=0.5)

        plt.grid()
        plt.legend()
        plt.savefig(path_here('TriangularParameterSampler_test_histogram_normalized.png'))
        plt.close()


        # Save a plot that shows the sample sequence
        plt.figure()
        plt.plot(appds, 'x')
        plt.grid()
        plt.savefig(path_here('TriangularParameterSampler_test_histogram_normalized_sequence.png'))
        plt.close()


class TestUniformAPPDSampler(unittest.TestCase):

    def setUp(self):
        # Load an image an image and its calibration parameters
        self.test_image = cv2.imread(path_here('test_image_before.png'))
        self.test_image_D = np.array([-0.2967039649743125, 0.06795775093662262, 0.0008927768064001824,
                                      -0.001327854648098482, 0.0])
        self.test_image_K = np.array([336.7755634193813, 0.0, 333.3575643300718, 0.0, 336.02729840829176,
                                      212.77376312080065, 0.0, 0.0, 1.0]).reshape((3, 3))
        self.width = self.test_image.shape[1]
        self.height = self.test_image.shape[0]
        self.ranges = { 'fx': (0.95*336.7755634193813, 1.05*336.7755634193813),
                        'fy': (0.95*336.02729840829176, 1.05*336.02729840829176),
                        'cx': (0.95*333.3575643300718, 1.05*333.3575643300718),
                        'cy': (0.95*212.77376312080065, 1.05*212.77376312080065),
                        'k1': (1.05*-0.2967039649743125, 0.95*-0.2967039649743125),
                        'k2': (0.95*0.06795775093662262, 1.05*0.06795775093662262),
                        'p1': (0.95*0.0008927768064001824, 1.05*0.0008927768064001824),
                        'p2': (1.05*-0.001327854648098482, 0.95*-0.001327854648098482),
                        'k3': (0.0, 0.0)}

        # Initialize a reference calibration object
        self.reference = Calibration(K=self.test_image_K, D=self.test_image_D,
                                     width=self.test_image.shape[1], height=self.test_image.shape[0])

    def test_histogram_normalized(self):

        plt.figure()

        # ParameterSampler
        sampler = ParameterSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height)
        t = time.time()
        bins, hist_edges = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES,
                                             n_bins=10, width=int(self.width/4), height=int(self.height/4), normalized=True,
                                             min_cropped_size=(int(self.width/4/1.5), int(self.height/4/1.5)))
        print("ParameterSampler time for %d normalized samples (1/16 res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge", label="ParameterSampler", alpha=0.5)

        # UniformAPPDSampler
        t = time.time()
        sampler = UniformAPPDSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height,
                                     reference=self.reference, temperature=5.0, appd_range_bins=10, init_jobs=4,
                                     appd_range_dicovery_samples=HISTOGRAM_TEST_SAMPLES,
                                     width=int(self.width / 4), height=int(self.height / 4),
                                     min_cropped_size=(int(self.width / 4 / 1.5), int(self.height / 4 / 1.5)))
        print("UniformAPPDSampler initialization time: %.02f" % (time.time()-t))

        t = time.time()
        bins, hist_edges, appds = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES, return_values=True,
                                                    n_bins=10, width=int(self.width/4), height=int(self.height/4),
                                                    normalized=True)
        print("UniformAPPDSampler time for %d normalized samples (1/16 res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge", label="UniformAPPDSampler", alpha=0.5)

        plt.grid()
        plt.legend()
        plt.savefig(path_here('UniformAPPDSampler_test_histogram_normalized.png'))
        plt.close()

        # Save a plot that shows the sample sequence
        plt.figure()
        plt.plot(appds, 'x')
        plt.grid()
        plt.savefig(path_here('UniformAPPDSampler_test_histogram_normalized_sequence.png'))
        plt.close()

    def test_ParallelBufferedSampler(self):

        plt.figure()

        # UniformAPPDSampler
        t = time.time()
        sampler = UniformAPPDSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height,
                                     reference=self.reference, temperature=5.0, appd_range_bins=10, init_jobs=4,
                                     appd_range_dicovery_samples=HISTOGRAM_TEST_SAMPLES,
                                     width=int(self.width / 4), height=int(self.height / 4),
                                     min_cropped_size=(int(self.width / 4 / 1.5), int(self.height / 4 / 1.5)))
        sampler = ParallelBufferedSampler(sampler=sampler, n_jobs=4, buffer_size=50)
        print("ParallelBufferedSampler with UniformAPPDSampler initialization time: %.02f" % (time.time()-t))

        t = time.time()
        bins, hist_edges, appds = sampler.histogram(self.reference, n_samples=HISTOGRAM_TEST_SAMPLES, return_values=True,
                                                    n_bins=10, width=int(self.width/4), height=int(self.height/4),
                                                    normalized=True)
        print("ParallelBufferedSampler with UniformAPPDSampler time for %d normalized samples (1/16 res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.bar(hist_edges[:-1], bins, width=np.diff(hist_edges), ec="k", align="edge", label="UniformAPPDSampler", alpha=0.5)
        sampler.stop()
        plt.grid()
        plt.legend()
        plt.savefig(path_here('ParallelBufferedSampler_UniformAPPDSampler_test_histogram_normalized_small.png'))
        plt.close()

        # Save a plot that shows the sample sequence
        plt.figure()
        plt.plot(appds, 'x')
        plt.grid()
        plt.savefig(path_here('ParallelBufferedSampler_UniformAPPDSampler_test_histogram_normalized_sequence.png'))
        plt.close()

    def test_hist_from_returned_objects(self):
        plt.figure()

        # UniformAPPDSampler
        t = time.time()
        sampler = UniformAPPDSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height,
                                     reference=self.reference, temperature=5.0, appd_range_bins=20,
                                     init_jobs=4, appd_range_dicovery_samples=HISTOGRAM_TEST_SAMPLES,
                                     width=int(self.width / 4), height=int(self.height / 4),
                                     min_cropped_size=(int(self.width / 4 / 1.5), int(self.height / 4 / 1.5)))
        sampler = ParallelBufferedSampler(sampler=sampler, n_jobs=4, buffer_size=50)
        print("ParallelBufferedSampler with UniformAPPDSampler initialization time: %.02f" % (time.time()-t))

        appds = list()
        t = time.time()
        for i in range(HISTOGRAM_TEST_SAMPLES):
            # The sampler will enforce the min correct size
            c = sampler.next()
            appds.append(c.appd(reference=self.reference, width=self.width, height=self.height, normalized=True))
            _ = c.rectify(image=self.test_image, mode='preserving')
        print("ParallelBufferedSampler with UniformAPPDSampler, APPD calculation and rectification for %d normalized samples (full res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.hist(appds, ec="k")
        sampler.stop()
        plt.grid()
        plt.savefig(path_here('ParallelBufferedSampler_UniformAPPDSampler_test_hist_from_returned_objects.png'))
        plt.close()

        # Save a plot that shows the sample sequence (useful for debugging bad random initialization)
        plt.figure()
        plt.plot(appds, 'x')
        plt.grid()
        plt.savefig(path_here('ParallelBufferedSampler_UniformAPPDSampler_test_hist_from_returned_objects_sequence.png'))
        plt.close()

    def test_compare_noncached_and_cached_ParallelBufferedSampler(self):
        plt.figure()

        # UniformAPPDSampler
        sampler = UniformAPPDSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height,
                                     reference=self.reference, temperature=5.0, appd_range_bins=20,
                                     init_jobs=4, appd_range_dicovery_samples=HISTOGRAM_TEST_SAMPLES,
                                     width=int(self.width / 4), height=int(self.height / 4),
                                     min_cropped_size=(int(self.width / 4 / 1.5), int(self.height / 4 / 1.5)))
        t = time.time()

        # NONCACHED -----------------------------------------
        noncached_sampler = ParallelBufferedSampler(sampler=sampler, n_jobs=4, buffer_size=50)
        print("Non-cached ParallelBufferedSampler with UniformAPPDSampler initialization time: %.02f" % (time.time()-t))

        appds = list()
        t = time.time()
        for i in range(2*HISTOGRAM_TEST_SAMPLES):
            # The sampler will enforce the min correct size
            c = noncached_sampler.next()
            appds.append(c.appd(reference=self.reference, width=self.width, height=self.height, normalized=True))
            _ = c.rectify(image=self.test_image, mode='preserving')
        print("Non-cached ParallelBufferedSampler with UniformAPPDSampler, APPD calculation and rectification for %d normalized samples (full res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.hist(appds, ec="k", label='non-cached, t=%ds' % int(time.time()-t))
        noncached_sampler.stop()

        # CACHED -----------------------------------------
        t = time.time()
        cached_sampler = ParallelBufferedSampler(sampler=sampler, n_jobs=1, buffer_size=5, cache_size=200)
        print("Cached ParallelBufferedSampler with UniformAPPDSampler initialization time: %.02f" % (time.time()-t))

        appds = list()
        t = time.time()
        for i in range(2*HISTOGRAM_TEST_SAMPLES):
            # The sampler will enforce the min correct size
            c = cached_sampler.next()
            appds.append(c.appd(reference=self.reference, width=self.width, height=self.height, normalized=True))
            _ = c.rectify(image=self.test_image, mode='preserving')
        print("Cached ParallelBufferedSampler with UniformAPPDSampler, APPD calculation and rectification for %d normalized samples (full res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.hist(appds, ec="k", label='cached, t=%ds' % int(time.time()-t))
        cached_sampler.stop()

        # ----------------------------------------------

        plt.grid()
        plt.legend()
        plt.savefig(path_here('non_cached_vs_cached_ParallelBufferedSampler.png'))
        plt.close()

        plt.figure()
        plt.plot(appds, 'x')
        plt.grid()
        plt.savefig(path_here('non_cached_vs_cached_ParallelBufferedSampler_sequence.png'))
        plt.close()

    def test_compare_noncached_and_cached_TriangularParallelBufferedSampler(self):
        plt.figure()

        # UniformAPPDSampler
        sampler = UniformAPPDSampler(ranges=self.ranges, cal_width=self.width, cal_height=self.height, sampler='triangular',
                                     reference=self.reference, temperature=5.0, appd_range_bins=20,
                                     init_jobs=4, appd_range_dicovery_samples=HISTOGRAM_TEST_SAMPLES,
                                     width=int(self.width / 4), height=int(self.height / 4),
                                     min_cropped_size=(int(self.width / 4 / 1.5), int(self.height / 4 / 1.5)))
        t = time.time()

        # # NONCACHED -----------------------------------------
        # noncached_sampler = ParallelBufferedSampler(sampler=sampler, n_jobs=4, buffer_size=50)
        # print("Non-cached TriangularParallelBufferedSampler with UniformAPPDSampler initialization time: %.02f" % (time.time()-t))
        #
        # appds = list()
        # t = time.time()
        # for i in range(2*HISTOGRAM_TEST_SAMPLES):
        #     # The sampler will enforce the min correct size
        #     c = noncached_sampler.next()
        #     appds.append(c.appd(reference=self.reference, width=self.width, height=self.height, normalized=True))
        #     _ = c.rectify(image=self.test_image, mode='preserving')
        # print("Non-cached TriangularParallelBufferedSampler with UniformAPPDSampler, APPD calculation and rectification for %d normalized samples (full res): %.02f, AVG: %.04f" %
        #       (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        # plt.hist(appds, ec="k", label='non-cached, t=%ds' % int(time.time()-t))
        # noncached_sampler.stop()

        # CACHED -----------------------------------------
        t = time.time()
        cached_sampler = ParallelBufferedSampler(sampler=sampler, n_jobs=1, buffer_size=5, cache_size=200)
        print("Cached TriangularParallelBufferedSampler with UniformAPPDSampler initialization time: %.02f" % (time.time()-t))

        appds = list()
        t = time.time()
        for i in range(2*HISTOGRAM_TEST_SAMPLES):
            # The sampler will enforce the min correct size
            c = cached_sampler.next()
            appds.append(c.appd(reference=self.reference, width=self.width, height=self.height, normalized=True))
            _ = c.rectify(image=self.test_image, mode='preserving')
        print("Cached TriangularParallelBufferedSampler with UniformAPPDSampler, APPD calculation and rectification for %d normalized samples (full res): %.02f, AVG: %.04f" %
              (HISTOGRAM_TEST_SAMPLES, time.time()-t, (time.time()-t)/HISTOGRAM_TEST_SAMPLES))
        plt.hist(appds, ec="k", label='cached, t=%ds' % int(time.time()-t))
        cached_sampler.stop()

        # ----------------------------------------------

        plt.grid()
        plt.legend()
        plt.savefig(path_here('non_cached_vs_cached_TriangularParallelBufferedSampler.png'))
        plt.close()

        plt.figure()
        plt.plot(appds, 'x')
        plt.grid()
        plt.savefig(path_here('non_cached_vs_cached_TriangularParallelBufferedSampler_sequence.png'))
        plt.close()


if __name__ == "__main__":
    unittest.main()
