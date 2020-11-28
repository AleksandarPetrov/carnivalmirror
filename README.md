# Carnival Mirror
[![PyPI version](https://badge.fury.io/py/carnivalmirror.svg)](https://badge.fury.io/py/carnivalmirror) [![Documentation Status](https://readthedocs.org/projects/carnivalmirror/badge/?version=stable)](https://carnivalmirror.readthedocs.io/en/stable/?badge=stable)

Carnival Mirror is a tool to simulate pinhole camera miscalibrations

# Installation
The recommended installation procedure is via pip:
```
pip install carnivalmirror
```

Alternatively, one can clone this repository and run `python setup.py install` in the main directory.

# Use
For an example of the common use of the package, see the `example.py` script.

# Tests
A number of unit tests are packaged. These can be run with `python setup.py test` or `python carnivalmirror/tests/tests.py`.

# Projects that use this library
Some examples of projects that use this library are:

- [Learning Camera Miscalibration Detection](https://github.com/ethz-asl/camera_miscalib_detection), an [ICRA 2020 paper](https://arxiv.org/abs/2005.11711) by the [Autonomous Systems Lab](https://asl.ethz.ch/) at ETH Zürich.
- The [Duckietown](duckietown.org) self-driving car [simulator environment for OpenAI Gym](https://github.com/duckietown/gym-duckietown)
