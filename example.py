"""
A simple example use of the CarnivalMirror library for generating miscalibrated images.
This shows the basic functionality and the main parameters, for more details consult the
documentation.
"""

import glob
import numpy as np

import cv2

import carnivalmirror as cm


# Hardcoded camera and distortion parameters
K = np.array([2304.54786556982, 0.000000, 1686.23787612802, 0.000000, 2305.875668062, 1354.98486439791, 0.000000, 0.000000, 1.000000]).reshape((3,3))
D = np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000])
width = 3384
height = 2710

# Create a Calibration object for the correct calibration. That will be used
# as a reference calibration when calculating APPD values
reference = cm.Calibration(K=K, D=D, width=width, height=height)

# Define ranges for the parameters:
# Note that if the ranges are too high and result in images with bad distortions
# (valid and cropped regions are too small or can't be defined), the sampling will be
# very slow.
ranges = {'fx': (0.95 * K[0,0], 1.05 * K[0,0]),
          'fy': (0.95 * K[1,1], 1.05 * K[1,1]),
          'cx': (0.95 * K[0,2], 1.05 * K[0,2]),
          'cy': (0.95 * K[1,2], 1.05 * K[1,2]),
          'k1': (-0.03, 0.03),
          'k2': (-0.003, 0.003),
          'p1': (-0.00002, 0.00002),
          'p2': (-0.0002, 0.0002),
          'k3': (-0.0001, 0.0001)}

# Create a Sampler object. Two samplers are provided:
#   ParameterSampler:   samples uniformly in the camera intrinsics and
#                       distortion parameters on a specified range
#   UniformAPPDSampler: samples camera intrinsics and distortion parameters
#                       such that the resulting APPD distribution is nearly
#                       uniform. Uses rejection sampling so is slower than
#                       ParameterSampler

# Create a ParameterSampler:
# sampler = cm.ParameterSampler(ranges=ranges, cal_width=width, cal_height=height)

# Create a UniformAPPDSampler
sampler = cm.UniformAPPDSampler(ranges=ranges, cal_width=width, cal_height=height, reference=reference,
                                temperature=5, appd_range_dicovery_samples=1000, appd_range_bins=10,
                                width=int(width/6), height=int(height/6),
                                min_cropped_size=(int(width/6/1.5), int(height/6/1.5)))

print("Initialized the sampler")

# Add parallelization and buffering to the sampler. This will create copies of
# the sampler that will run on different threads and will fill in the same buffer
# with calibrations. Parallelization and buffering can significantly speed up the
# sampling, particularly when using the UniformAPPDSampler.

sampler = cm.ParallelBufferedSampler(sampler=sampler, buffer_size=16, n_jobs=4)
print("Added parallelization and buffering to the sampler")
# Now we are ready to load the images and rectify them with sampled calibration
# parameters. We will generate 5 images for every input image.

images_path = 'example_images'
for file in glob.glob(images_path + "/*_Camera_5.jpg"):
    print("Generating images for file %s" % file)
    image = cv2.imread(file)

    for i in range(5):

        # Sample a set of calibration parameters
        c = sampler.next()

        # Calculate the APPD value for these parameters
        appd, diff_map = c.appd(reference=reference, width=width, height=height,
                                return_diff_map=True, normalized=True)

        # (Mis)rectify the image
        # Two modes can be used:
        #   standard:   as is, without cropping and rescaling
        #   preserving: with cropped and rescaled vaild aspect-ratio-preserving region
        new_image = c.rectify(image, result_width=width, result_height=height, mode='preserving')

        # Show the APPD value on top of the image
        cv2.putText(new_image, 'APPD %.5f'%appd, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 3)

        cv2.imwrite(file.split('.jpg')[0]+"_gen_%d"%(i+1)+".jpg", new_image)



# Do that to stop the background sampling
sampler.stop()



