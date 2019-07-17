"""Definition of the various Sampler objects"""

import time
import copy
import multiprocessing
import queue

import numpy as np

from .calibration import Calibration

class Sampler(object):
    """Base Sampler class

        Attributes:
            cal_width (:obj:`int`): The width of the image(s) for which the calibrations are
            cal_height (:obj:`int`): The height of the image(s) for which the calibrations are
            aspect_ratio (:obj:`float`): The calibration aspect ratio.
    """

    def __init__(self, cal_width, cal_height):
        """Initializes a Sampler object

        Args:
            cal_width (:obj:`int`): The width of the image(s) for which the calibrations are
            cal_height (:obj:`int`): The height of the image(s) for which the calibrations are
        """
        self.cal_width = cal_width
        self.cal_height = cal_height
        self.aspect_ratio = cal_width / cal_height

    def next(self):
        """Should be implemented by children classes"""
        raise NotImplementedError

    def histogram(self, reference, n_samples=1000, n_bins=10, return_values=False, **kwargs):
        """Obtains a histogram of APPD values

        Samples n_samples times from the ranges of the parameters, calculates their
        APPD values with respect to a reference Calibration and builds a histogram from them.

        Args:
            reference (:obj:`Calibration`): A reference Calibration object
            n_samples (:obj:`int`): Number of samples
            n_bins (:obj:`int`): Number of histogram bins
            return_values (:obj:`bool`): If `True`, will also return the sampled APPD values
            **kwargs: Arguments to be passed to the `appd` method of Calibration (width and height required)

        Returns:
            tuple: a float or a tuple containing:

                hist (:obj:`numpy array`): The values of the histogram (number of occurrences)

                bin_edges (:obj:`numpy array`): The bin edges `(length(hist)+1)`

                appd_values (:obj:`list` of :obj:`float`): The sampled values (only if `return_values` is set to `True`)

        """

        APPD_values = list()

        for _ in range(n_samples):
            sample_obtained = False
            while not sample_obtained:
                try:
                    APPD_values.append(self.next().appd(reference=reference, **kwargs))
                    sample_obtained = True
                except RuntimeError as e:
                    # If the parameter set is bad (no valid region or too small) sample again
                    pass
        if return_values:
            r = np.histogram(APPD_values, bins=n_bins)
            return r[0], r[1], APPD_values
        else:
            return np.histogram(APPD_values, bins=n_bins)

class ParallelBufferedSampler(Sampler):
    """ParallelBufferedSampler paralelizes and buffers a given sampler.

    Creates several background treads that sample :class:`~.Calibration` objects and maintain a
    buffer of a specified size.

    In order to stop the threads (and gracefully exit the parent process) you
    have to use the :meth:`~.ParallelBufferedSampler.stop` method.

    """

    def __init__(self, sampler, n_jobs=4, buffer_size=16):
        """Given an initialized sampler, abstracts its parallelization and buffering.

        Args:
            sampler (:obj:`Sampler`): A Sampler object to parallelize
            n_jobs (:obj:`int`): Number of threads to create
            buffer_size (:obj:`int`): Number of calibrations to keep in the buffer

        """

        self.init_sampler = sampler
        self.n_jobs = n_jobs
        self.buffer_size = buffer_size

        # Copy all the attributes of the sampler to the ParallelBufferedSampler
        self.__dict__ = sampler.__dict__.copy()

        # Make a copy of the sampler for every thread
        self.samplers = list()
        for _ in range(n_jobs): self.samplers.append(copy.deepcopy(sampler))

        # Create the buffer Queue
        self.bufferQueue = multiprocessing.Queue(maxsize=buffer_size)
        self.stopTrigger = multiprocessing.Event()

        # The function that each thread runs
        def thread_fun(sampler, buffer, stop):
            # Make sure each copy has a unique numpy seed
            np.random.seed(multiprocessing.current_process().pid)

            # Keep filling the buffer until requested to stop
            # Convoluted because we want to catch when 'stop' becomes set
            # and at the same time we don't want to sample unnecessarily
            new_sample = None
            while not stop.is_set():
                if new_sample is None:
                    new_sample = sampler.next()
                try:
                    buffer.put(new_sample, block=True, timeout=0.5)
                    new_sample = None
                except queue.Full:
                    pass

        # Start the threads
        self.processes = list()
        for i in range(n_jobs):
            p = multiprocessing.Process(target=thread_fun, args=(self.samplers[i], self.bufferQueue, self.stopTrigger, ))
            p.start()
            self.processes.append(p)

    def next(self):
        """Provide the next :obj:`Calibration` from the buffer.

        Returns:
            :obj:`Calibration`: A :obj:`Calibration` object
        """
        return self.bufferQueue.get(block=True)

    def stop(self):
        """Stops the background processes that fill the buffer."""
        self.stopTrigger.set()

        # Give the threads a chance to shut off properly
        time.sleep(1.5)
        for i in range(len(self.processes)):
            self.processes[i].terminate()


class ParameterSampler(Sampler):
    """ParameterSampler provides uniform independent sampling within specified ranges of
    calibration parameters.

    Attributes:
        width (:obj:`int`): The width of the image(s) for which the calibrations are
        height (:obj:`int`): The height of the image(s) for which the calibrations are
        aspect_ratio (:obj:`float`): The calibration aspect ratio.
        ranges (:obj:`dict`): The provided sampling ranges for the calibration parameters
    """

    def __init__(self, ranges, cal_width, cal_height):
        """Initializes a ParameterSampler object

        Args:
            ranges (:obj:`dict`): A dictionary with keys `[fx, fy, cx, cy, k1, k2, p1, p2, k3]` and elements tuples
                describing the sampling range for each parameter. All intrinsic parameters must be provided.
                Missing distortion parameters will be sampled as 0
            cal_width (:obj:`int`): The width of the image(s) for which the calibrations are
            cal_height (:obj:`int`): The height of the image(s) for which the calibrations are
        Raises:
            ValueError: If one of `[fx, fy, cx, cy]` is missing from `ranges`
        """

        super(ParameterSampler, self).__init__(cal_width=cal_width, cal_height=cal_height)

        # Validate the ranges
        for key in ['fx', 'fy', 'cx', 'cy']:
            if key not in ranges: raise ValueError("Key %s missing in ranges" % key)
        for key in ['k1', 'k2', 'p1', 'p2', 'k3']:
            if key not in ranges: ranges[key] = (0, 0)
        self.ranges = ranges

    def next(self):
        """Generator method providing a randomly sampled :obj:`Calibration`

        Returns:
            :obj:`Calibration`: A :obj:`Calibration` object
        """

        # Sample the values
        sample = dict()
        for key in self.ranges:
            sample[key] = np.random.uniform(self.ranges[key][0], self.ranges[key][1])

        # Construct a Calibration object
        K = np.array([[sample['fx'],    0.0,            sample['cx']],
                      [0.0,             sample['fy'],   sample['cy']],
                      [0.0,             0.0,            1.0         ]])
        D = np.array([sample['k1'], sample['k2'], sample['p1'], sample['p2'], sample['k3']])

        return Calibration(K=K, D=D, width=self.cal_width, height=self.cal_height)


class UniformAPPDSampler(Sampler):
    """UniformAPPDSampler provides parameter sampling that results in approximately uniform APPD distribution

        Attributes:
            width (:obj:`int`): The width of the image(s) for which the calibrations are
            height (:obj:`int`): The height of the image(s) for which the calibrations are
            aspect_ratio (:obj:`float`): The calibration aspect ratio.
            ranges (:obj:`dict`): The provided sampling ranges for the calibration parameters
            temperature (:obj:`float`): Temperature used for Gibbs acceptance sampling
            reference (:obj:`Calibration`): The reference :obj:`Calibration`
            bin_edges (:obj:`list` of :obj:`float`): Marks the bin edges
            bin_counts (:obj:`list` of :obj:`int`): Stores how many samples were generated from each bin so far
    """

    def __init__(self, ranges, cal_width, cal_height, reference, temperature=1, appd_range_dicovery_samples=1000,
                 appd_range_bins=10, init_jobs=1, **kwargs):
        """Initializes a UniformAPPDSampler object

        Args:
            ranges (:obj:`dict`): A dictionary with keys `[fx, fy, cx, cy, k1, k2, p1, p2, k3]` and elements tuples
                describing the sampling range for each parameter. All intrinsic parameters must be provided.
                Missing distortion parameters will be sampled as 0
            cal_width (:obj:`int`): The width of the image(s) for which the calibrations are
            cal_height (:obj:`int`): The height of the image(s) for which the calibrations are
            reference (:obj:`Calibration`): A reference :class:`~.Calibration` object
            temperature (:obj:`float`): Temperature used for Gibbs acceptance sampling
            appd_range_dicovery_samples (:obj:`int`): Number of samples obtained in order to find the
                range of achievable APPD values
            appd_range_bins (:obj:`int`): Number of histogram bins
            init_jobs (:obj:`int`): Number of jobs used for the initialization sampling
            **kwargs: Arguments to be passed to the appd method of :class:`~.Calibration` (`width` and `height`
                required, optionally `map_width`, `map_height`, `interpolation`)
        Raises:
            ValueError: If one of `[fx, fy, cx, cy]` is missing from ranges
            RuntimeException: If a histogram bin with zero elements is encountered.
        """

        super(UniformAPPDSampler, self).__init__(cal_width=cal_width, cal_height=cal_height)

        self.temperature = temperature
        self.reference = reference
        self.kwargs = kwargs

        # Validate the ranges
        for key in ['fx', 'fy', 'cx', 'cy']:
            if key not in ranges: raise ValueError("Key %s missing in ranges" % key)
        for key in ['k1', 'k2', 'p1', 'p2', 'k3']:
            if key not in ranges: ranges[key] = (0, 0)
        self.ranges = ranges

        # Find the approximate range of normalized APPD values achievable by the ranges provided by sampling
        ps = ParameterSampler(ranges=self.ranges, cal_width=self.cal_width, cal_height=self.cal_height)
        # Parallelize if requested
        if init_jobs > 1:
            ps = ParallelBufferedSampler(sampler=ps, n_jobs=init_jobs, buffer_size=init_jobs*4)
        hist, bin_edges = ps.histogram(reference=reference, n_samples=appd_range_dicovery_samples,
                                       n_bins=appd_range_bins, normalized=True, **kwargs)
        if init_jobs > 1:
            ps.stop()

        # Make sure that all bins have at least 1 sample. This ensures that all bins can be sampled and the procedure
        # won't end up in a deadlock
        for b_idx, b in enumerate(hist):
            if not b > 0:
                bin_edges = bin_edges[:b_idx+1]

        # Make the attributes that store the bins
        self.APPDrange = (bin_edges[0], bin_edges[-1])
        self.bin_edges = np.array(bin_edges)
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        self.bin_counts = np.array([0]*(len(bin_edges)-1))


    def next(self):
        """Generator method providing a randomly sampled Calibration that results in a
        nearly uniform APPD distribution

        Samples are taken uniformly but are accepted with probability `-diff_with_smallest_bin/temperture`.

        Returns:
            :obj:`Calibration`: A :obj:`Calibration` object
        """

        # Calculate the current acceptance probabilities
        accept_probs = np.exp(-(self.bin_counts-np.min(self.bin_counts))/self.temperature)
        # print('accept_probs')
        # for i in range(len(accept_probs)):
        #     print(self.bin_edges[i+1], accept_probs[i], self.bin_counts[i])
        # print('sampled so far', np.sum(self.bin_counts))

        # Sample until we accept
        accepted = False
        while not accepted:

            # Sample the values until a valid rectification is obtained
            sample_obtained = False
            while not sample_obtained:

                sample = dict()
                for key in self.ranges:
                    sample[key] = np.random.uniform(self.ranges[key][0], self.ranges[key][1])

                # Construct a Calibration object
                K = np.array([[sample['fx'], 0.0, sample['cx']],
                              [0.0, sample['fy'], sample['cy']],
                              [0.0, 0.0, 1.0]])
                D = np.array([sample['k1'], sample['k2'], sample['p1'], sample['p2'], sample['k3']])

                c = Calibration(K=K, D=D, width=self.cal_width, height=self.cal_height)

                # Calculate the APPD for this calibration
                try:
                    appd = c.appd(self.reference, normalized=True, **self.kwargs)
                    sample_obtained = True
                except RuntimeError as e:
                    # If the parameter set is bad (no valid region or too small) sample again
                    pass

            # Take a chance on accepting
            for i in range(len(self.bin_edges)-1):
                if self.bin_edges[i+1]>appd:
                    appd_in_bin = i
                    break

            if np.random.rand() < accept_probs[appd_in_bin]:
                accepted = True
                self.bin_counts[appd_in_bin] += 1

        return c





