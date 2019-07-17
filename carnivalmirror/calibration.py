"""Definition of the Calibration object"""

import copy

import numpy as np
import cv2

class Calibration(object):
    """The Calibration object represents a set of intrinsic and distortion
    parameters and enables basic computation on them.

    Attributes:
        n_fx (:obj:`float`): Normalized fx intrinsic camera parameter (for height 1 pixel).
        n_fy (:obj:`float`): Normalized fy intrinsic camera parameter (for height 1 pixel).
        n_cx (:obj:`float`): Normalized cx intrinsic camera parameter (for height 1 pixel).
        n_cy (:obj:`float`): Normalized cy intrinsic camera parameter (for height 1 pixel).
        k1 (:obj:`float`): The k1 radial distortion parameter.
        k2 (:obj:`float`): The k2 radial distortion parameter.
        k3 (:obj:`float`): The k3 radial distortion parameter.
        p1 (:obj:`float`): The p1 tangential distortion parameter.
        p2 (:obj:`float`): The p2 tangential distortion parameter.
        aspect_ratio (:obj:`float`): The calibration aspect ratio.

    """

    def __init__(self, K, D, width, height):
        """Initializes a Calibration object.

        Initializes a calibration object from a set of intrinsic parameters
        and distortion coefficients.

        Args:
            K: A list with the camera matrix values ordered as `[fx, fy, cx, cy]` or
                a 3x3 array representing a camera matrix
            D: A list of distortion coefficients ordered as `[k1, k2, p1, p2, k3]`, all 5
                coefficients should be provided
            width (:obj:`int`): Width of the image resolution at which the camera matrix was obtained
            height (:obj:`int`): Height of the image resolution at which the camera matrix was obtained

        Raises:
            TypeError: If any of the inputs do not match the expected types or dimensions.

        """

        # Normalize and save the inputs
        self.aspect_ratio = float(width) / height
        K = np.array(K)
        if len(K.flatten()) == 4:
            fx, fy, cx, cy = (K[0], K[1], K[2], K[3])
        elif K.shape == (3, 3):
            fx, fy, cx, cy = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        else:
            raise TypeError("Input K was not a list of length 4 or a 3x3 matrix")

        # Normalize all by the height so that we and with an image with height 1 and
        # width equal to the aspect ratio self.ar
        self.n_fx = float(fx) / height
        self.n_cx = float(cx) / height
        self.n_fy = float(fy) / height
        self.n_cy = float(cy) / height

        D = np.array(D).flatten()
        if len(D) == 5:
            self.k1, self.k2, self.p1, self.p2, self.k3 = (D[0], D[1], D[2], D[3], D[4])
        else:
            raise TypeError("Input D was not a list of length 5")

        # Initialize some properties
        self._size_used_for_calculations = None
        self.new_camera_matrix = None
        self.valid_region = None
        self.undistort_x = None
        self.undistort_y = None

    def __repr__(self):
        str = "Calibration with:\n"
        str+= "    \t[%.2f\t%.2f\t%.2f]\n" % (self.n_fx, 0, self.n_cx)
        str+= "K = \t|%.2f\t%.2f\t%.2f]\n" % (0, self.n_fy, self.n_cy)
        str+= "    \t[%.2f\t%.2f\t%.2f]\n\n" % (0, 0, 1)
        str+= "D = \t[%.2f\t%.2f\t%.2f\t%.2f\t%.2f]" % (self.k1, self.k2, self.p1, self.p2, self.k3)
        return str

    def get_K(self, height=1):
        """Calculates the camera matrix for a particular resolution.

        Calculates the camera matrix for an image with height (in pixels) as supplied,
        and width equal to `self.aspect_ratio*height`.

        Args:
            height (:obj:`int`): An interger representing the desired height (in pixels)
                for the camera matrix
        Returns:
            numpy array: A 3x3 camera matrix for the desired resolution
        """

        fx = self.n_fx * height
        fy = self.n_fy * height
        cx = self.n_cx * height
        cy = self.n_cy * height

        return np.array([[fx , 0.0, cx ],
                         [0.0, fy,  cy ],
                         [0.0, 0.0 ,1.0]])

    def get_D(self):
        """Provides the distortion coefficents.

        Returns:
            numpy array: A list of the 5 distortion coefficients `[k1, k2, p1, p2, k3]`
        """
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    def undistortion_maps(self, width, height, map_width=None, map_height=None, interpolation=cv2.INTER_LANCZOS4):
        """Calculates the undistortion maps.

        Based on OpenCV's solution:
        https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html

        Args:
            width (:obj:`int`): Desired width for the input image
            height (:obj:`int`): Desired height for the input image
            map_width (:obj:`int`): Desired width for the undistortion maps,
                default None and will use the input width
            map_height (:obj:`int`): Desired height for the undistortion maps,
                default None and will use the input height
            interpolation: cv2 interpolation method, default is `cv2.INTER_LANCZOS4`

        Returns:
            (numpy array, numpy array): The undistortion maps for the x-axis and y-axis

        Raises:
            ValueError: If only one of `map_width` and `map_height` is `None`
        """

        # Calculate the undistorion maps, only if the previous were for a different resolution
        if (width, height) != self._size_used_for_calculations:
            # Calculate the new camera matrix
            self.new_camera_matrix, self.valid_region = cv2.getOptimalNewCameraMatrix(cameraMatrix=self.get_K(height=height),
                                                                                   distCoeffs=self.get_D(),
                                                                                   imageSize=(width, height),
                                                                                   alpha=1)

            # Calculate the undistorting maps
            self.undistort_x, self.undistort_y = cv2.initUndistortRectifyMap(cameraMatrix=self.get_K(height=height),
                                                                             distCoeffs=self.get_D(),
                                                                             R=np.eye(3),
                                                                             newCameraMatrix=self.new_camera_matrix,
                                                                             size=(width, height),
                                                                             m1type=cv2.CV_32FC1)

            # Update the stored height
            self._size_used_for_calculations = (width, height)


        # Resize them to the target output
        if map_width is None and map_height is None:
            resized_map_x = self.undistort_x
            resized_map_y = self.undistort_y
        elif map_width is not None and map_height is not None:
            resized_map_x = cv2.resize(self.undistort_x, (map_width, map_height), interpolation=interpolation)
            resized_map_y = cv2.resize(self.undistort_y, (map_width, map_height), interpolation=interpolation)
        else:
            raise ValueError("Both map_width and map_height should be None or numbers")


        return resized_map_x, resized_map_y

    def undistortion_maps_preserving(self, width, height, min_cropped_size=None, map_width=None,
                                     map_height=None, interpolation=cv2.INTER_LANCZOS4):
        """Calculates the preserving undistortion maps.

        Calculates an undistorion map that crops only the part of the undistorted image that has no missing
        pixels due to the rectification and that is the same aspect ratio as the provided size. Then it
        resizes it up to the original size.

        Args:
            width (:obj:`int`): The desired width for the input image
            height (:obj:`int`): The desired height for the input image
            min_cropped_size (:obj:`int`): Optional, if provided as a `(width, height)` tuple, will raise `RuntimeError` if the
                cropped image is smaller than this
            map_width (:obj:`int`): The desired width for the undistortion maps,
                if `None` will use the input width
            map_height (:obj:`int`): The desired height for the undistortion maps,
                if `None` will use the input height
            interpolation: cv2 interpolation method

        Returns:
            (numpy array, numpy array): The preserving undistortion maps for the x-axis and y-axis

        Raises:
            RuntimeError: If the region of validity is only zeros, if the cropped image
                is smaller than `min_cropped_size`, or if the aspect ratio cannot be preserved
            ValueError: If only one of `map_width` and `map_height` is None
        """

        # Make sure the maps are calculated for the right resolution
        self.undistortion_maps(width=width, height=height)

        # Make sure that the OpenCV valid region is actually valid
        if np.allclose(self.valid_region, 0, rtol=1e-04, atol=1e-07):
            raise RuntimeError("The region of validity is zero")


        # Find how to crop the new image in order to use only the region of correct
        # rectification while maintaining the original aspect ratio
        target_aspect_ratio = float(width) / float(height)

        valid_dict = {'x': (self.valid_region[0], self.valid_region[0] + self.valid_region[2]),
                    'y': (self.valid_region[1], self.valid_region[1] + self.valid_region[3])}
        valid_aspect_ratio = float(valid_dict['x'][1] - valid_dict['x'][0]) / (valid_dict['y'][1] - valid_dict['y'][0])

        # If the image is taller than it should, cut its legs and head
        if valid_aspect_ratio < target_aspect_ratio:
            desired_number_of_rows = int(round((valid_dict['x'][1] - valid_dict['x'][0]) / target_aspect_ratio))
            cut_top = int(((valid_dict['y'][1] - valid_dict['y'][0]) - desired_number_of_rows) / 2.0)
            cropped_region = {
                'x': valid_dict['x'],
                'y': (valid_dict['y'][0] + cut_top, valid_dict['y'][0] + cut_top + desired_number_of_rows)
            }
        else:
            desired_number_of_cols = int(round((valid_dict['y'][1] - valid_dict['y'][0]) * target_aspect_ratio))
            cut_left = int(((valid_dict['x'][1] - valid_dict['x'][0]) - desired_number_of_cols) / 2.0)
            cropped_region = {
                'x': (valid_dict['x'][0] + cut_left, valid_dict['x'][0] + cut_left + desired_number_of_cols),
                'y': valid_dict['y']
            }

        # Make sure the cropped aspect ratio is the same as the target one
        if cropped_region['x'][1] - cropped_region['x'][0] < 10 or (cropped_region['y'][1] - cropped_region['y'][0]) < 10:
            raise RuntimeError("The cropped region is too small: %s" % str(cropped_region))

        cropped_aspect_ratio = float(cropped_region['x'][1] - cropped_region['x'][0]) / (cropped_region['y'][1] - cropped_region['y'][0])
        if np.abs(cropped_aspect_ratio-target_aspect_ratio)>0.1:
            raise RuntimeError("The aspect ratio could not be preserved. Target: %.05f, Cropped: %.05f" %
                               (target_aspect_ratio, cropped_aspect_ratio))

        # Check if the cropped image fits the minimum requirements
        if min_cropped_size is not None and \
            (cropped_region['x'][1] - cropped_region['x'][0] < min_cropped_size[0] \
            or cropped_region['y'][1] - cropped_region['y'][0] < min_cropped_size[1]):
            raise RuntimeError("The cropped image has dimensions %dx%d, when the minimum required is %dx%d" % \
                               (int(cropped_region['x'][1] - cropped_region['x'][0]), int(cropped_region['y'][1] - cropped_region['y'][0]),
                                min_cropped_size[0], min_cropped_size[1]))

        # Set the output resolution
        if map_width is None and map_height is None:
            output_resolution = (width, height)
        elif map_width is not None and map_height is not None:
            output_resolution = (map_width, map_height)
        else:
            raise ValueError("Both map_width and map_height should be None or numbers")

        # Crop and resize the undistortion maps
        # TODO must check if the inputs and the maps are in the dimensions I expect them to be
        preserving_undistort_x = copy.deepcopy(self.undistort_x)
        preserving_undistort_y = copy.deepcopy(self.undistort_y)
        preserving_undistort_x = preserving_undistort_x[cropped_region['y'][0]:cropped_region['y'][1],
                                 cropped_region['x'][0]:cropped_region['x'][1]]
        preserving_undistort_x = cv2.resize(preserving_undistort_x, output_resolution, interpolation=interpolation)
        preserving_undistort_y = preserving_undistort_y[cropped_region['y'][0]:cropped_region['y'][1],
                                 cropped_region['x'][0]:cropped_region['x'][1]]
        preserving_undistort_y = cv2.resize(preserving_undistort_y, output_resolution, interpolation=interpolation)

        return preserving_undistort_x, preserving_undistort_y

    def rectify(self, image, interpolation=cv2.INTER_LANCZOS4, quiet=False, strict=False, result_width=None,
                result_height=None, mode='standard', min_cropped_size=None):
        """Rectify an image.

        Args:
            image (:obj:`numpy array`): The input image. Should be `cv2`-compatible. Should be `[width, height, channels]`
            interpolation: cv2 interpolation method
            quiet (:obj:`bool`): If `True`, won't show a warning if the aspect ratio of the image is not the same
                as the one of the calibration given during the initialization
            strict (:obj:`bool`): If `True`, will raise a `ValueError` exception if there is aspect ratio mismatch
            result_width (:obj:`int`): Desired width for the rectified image, if `None` will use the input height
            result_height (:obj:`int`): Desired height for the rectified image, if `None` will use the input height
            mode (:obj:`str`): Can be `standard` or `preserving`, chooses between standard undistortion (with `undistortion_maps`) and
                undistortion with cropped and rescaled vaild aspect-ratio-preserving region
                (with `undistortion_maps_preserving`).
            min_cropped_size (:obj:`tuple` of :obj:`int`): Passed to `undistortion_maps_preserving` if
                `mode` is set to `preserving`

        Returns:
            numpy array: The resulting image after rectification

        Raises:
            ValueError: See information about the `strict` parameter. Also raised if only one of `map_width` and
                `map_height` is `None`
        """
        image = np.array(image)
        image_height = image.shape[0]
        image_width = image.shape[1]
        image_ar = float(image_width)/image_height

        # Check if the aspect ratios match
        if not quiet and not np.isclose(image_ar, self.aspect_ratio, rtol=1e-04, atol=1e-07):
            msg = "The aspect ratio of the image is not the same as the " + \
                  "one of the calibration given during the initialization"
            if strict: raise ValueError(msg)
            else: print("WARNING: " + msg)

        # Set the output resolution
        if result_width is None and result_height is None:
            output_width, output_height = (image_width, image_height)
        elif result_width is not None and result_height is not None:
            output_width, output_height = (result_width, result_height)
        else:
            raise ValueError("Both image_width and image_height should be None or numbers")

        # Rectify with the desired function
        if mode == 'standard':
            map_x, map_y = self.undistortion_maps(height=image_height, width=image_width, interpolation=interpolation,
                                                  map_width=output_width, map_height=output_height)
        elif mode == 'preserving':
            map_x, map_y = self.undistortion_maps_preserving(height=image_height, width=image_width,
                                                             min_cropped_size=min_cropped_size,
                                                             interpolation=interpolation,
                                                             map_width=output_width, map_height=output_height)
        else:
            raise ValueError("Mode %s is not recognized." % str(mode))

        remappedIm = cv2.remap(image, map_x, map_y, interpolation)

        return remappedIm

    def appd(self, reference, width, height, interpolation=cv2.INTER_LANCZOS4,
             min_cropped_size=None, return_diff_map=False, map_width=None, map_height=None, normalized=False):
        """Calculate the Average Pixel Position Difference.

        Args:
            reference (:obj:`Calibration`): Another Calibration object relative to which the metric will be computed
            width (:obj:`int`): Desired width for the rectified image, a base for calculating the metric
            height (:obj:`int`): Desired height for the rectified image, a base for calculating the metric
            interpolation: `cv2` interpolation method
            quiet (:obj:`bool`): If `True`, won't show a warning if the aspect ratio of the image is not the same
                as the one of the calibration given during the initialization
            strict (:obj:`bool`): If `True`, will raise a `ValueError` exception if there is aspect ratio mismatch
            min_cropped_size (:obj:`tuple` of :obj:`int`): Passed to `undistortion_maps_preserving`
            return_diff_map (:obj:`bool`): If `True`, returns the difference map
            map_width (:obj:`int`): Width for the undistortion map, if `None`, `width`is used. Use the same aspect ratio
                for the maps as for the rectified image
            map_height (:obj:`int`): Height for the undistortion map, if `None`, `height` is used. Use the same aspect ratio
                for the maps as for the rectified image
            normalized (:obj:`bool`): If `True`, normalizes by the map diagonal

        Returns:
            tuple: a float or a tuple containing:

                appd (:obj:`float`): The Average Pixel Position Difference between this calibration and the reference,
                calculated for the provided resolution.

                diff_map (:obj:`numpy array`): If `return_diff_map` is `True` also returns the difference map

        Raises:
            ValueError: See information about the `strict` parameter
            RuntimeError: See information for `undistortion_maps_preserving`

        """

        if map_height == None: map_height=height
        if map_width == None: map_width=width

        # Obtain the rectification maps for this and the reference
        this_map_x, this_map_y = self.undistortion_maps_preserving(width=width, height=height,
                                                                   min_cropped_size=min_cropped_size,
                                                                   interpolation=interpolation,
                                                                   map_width=map_width,
                                                                   map_height=map_height)

        ref_map_x, ref_map_y = reference.undistortion_maps_preserving(width=width, height=height,
                                                                      min_cropped_size=min_cropped_size,
                                                                      interpolation=interpolation,
                                                                      map_width=map_width,
                                                                      map_height=map_height)

        # Calculate the difference maps
        diff_map_x = this_map_x-ref_map_x
        diff_map_y = this_map_y-ref_map_y
        diff_map = np.sqrt(np.square(diff_map_x)+np.square(diff_map_y))

        # Find the mean
        appd = np.mean(diff_map)

        # Apply normalization if requested
        if normalized:
            appd /= np.sqrt(map_width**2 + map_height**2)

        if return_diff_map:
            return appd, diff_map
        else:
            return appd
