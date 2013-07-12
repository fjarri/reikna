*************************************
Reikna, the pure Python GPGPU library
*************************************

|documentation_type_subtitle|

``Reikna`` is a library containing various GPU algorithms built on top of `PyCuda <http://documen.tician.de/pycuda>`_ and `PyOpenCL <http://documen.tician.de/pyopencl>`_.
The main design goals are:

* separation of computation cores (matrix multiplication, random numbers generation etc) from simple transformations on their input and output values (scaling, typecast etc);
* separation of the preparation and execution stage, maximizing the performance of the execution stage at the expense of the preparation stage (in other words, aiming at large simulations)
* partial abstraction from Cuda/OpenCL

``Reikna`` is hosted `on GitHub <http://github.com/Manticore/reikna>`_, issues should be filed there too.

********
Contents
********

.. toctree::
   :maxdepth: 2

   introduction
   tutorial-modules
   tutorial-basic
   tutorial-advanced
   correlations
   api/index
   history

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
