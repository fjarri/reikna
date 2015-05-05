=====================================
Reikna, the pure Python GPGPU library
=====================================

``Reikna`` is a library containing various GPU algorithms built on top of `PyCUDA <http://documen.tician.de/pycuda>`_ and `PyOpenCL <http://documen.tician.de/pyopencl>`_.
The main design goals are:

* separation of computation cores (matrix multiplication, random numbers generation etc) from simple transformations on their input and output values (scaling, typecast etc);
* separation of the preparation and execution stage, maximizing the performance of the execution stage at the expense of the preparation stage (in other words, aiming at large simulations)
* partial abstraction from CUDA/OpenCL

Tests can be run by installing `Py.Test <http://pytest.org>`_ and running ``py.test`` from the ``test`` folder (run ``py.test --help`` to get the list of options).

For more information proceed to the `project documentation page <http://reikna.publicfields.net>`_. If you have a general question that does not qualify as an issue, you can ask it at the `discussion forum <https://groups.google.com/d/forum/reikna>`_.

