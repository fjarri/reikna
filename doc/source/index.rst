***********************************
Reikna, a pure Python GPGPU library
***********************************

``Reikna`` is a library containing various GPU algorithms built on top of `PyCUDA <http://documen.tician.de/pycuda>`_ and `PyOpenCL <http://documen.tician.de/pyopencl>`_.
The main design goals are:

* separation of computation cores (matrix multiplication, random numbers generation etc) from simple transformations on their input and output values (scaling, typecast etc);
* separation of the preparation and execution stage, maximizing the performance of the execution stage at the expense of the preparation stage (in other words, aiming at large simulations)
* partial abstraction from CUDA/OpenCL

The installation is as simple as

::

    $ pip install reikna

Community resources
===================

* `Source repository <http://github.com/fjarri/reikna>`_ on GitHub;

* `Issue tracker <http://github.com/fjarri/reikna/issues>`_, *ibid.*;

* `Discussion forum <https://groups.google.com/d/forum/reikna>`_ on Google Groups.

********
Contents
********

.. toctree::
   :maxdepth: 2

   introduction
   tutorial-modules
   tutorial-basic
   tutorial-advanced
   api/index
   history

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
