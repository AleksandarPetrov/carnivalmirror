.. CarnivalMirror documentation master file, created by
   sphinx-quickstart on Mon Jul 15 19:48:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CarnivalMirror's documentation!
==========================================

CarnivalMirror is a Python library that can be used to simulate pinhole camera miscalibrations.


Features
--------

- Pinhole camera model with radial and tangential distortion
- Image rectification and misrectification
- Parallel sampling of miscalibration, either uniform in the parameters or approximately uniform in the resulting Average Pixel Position Difference
- A number of other useful tools

Installation
------------

The recommended installation procedure for CarnivalMirror is via pip:

::

    pip install carnivalmirror

Alternatively, one can clone this repository and run `python setup.py install` in the main directory.

Get started
-----------
For an example of the common use of the package, see the `example.py` script here: https://github.com/AleksandarPetrov/carnivalmirror .

Contribute
----------

- Source Code: https://github.com/AleksandarPetrov/carnivalmirror

Support
-------

If you are having issues, please let us know.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   index
   carnivalmirror



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
