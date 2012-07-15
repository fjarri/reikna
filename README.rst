=====================================
Tigger, the pure Python GPGPU library
=====================================

Tigger is a library containing various GPU algorithms.
The main design goals are:

* separation of computation cores (matrix multiplication, random numbers generation etc) from simple transformations on their input and output values (scaling, typecast etc);
* separation of the preparation and execution stage, maximizing the performance of the execution stage at the expense of the preparation stage (in other words, aiming at large simulations)
* partial abstraction from Cuda/OpenCL

Additional long-term goal is the separation of the kernel rendering stage and the actual usage of the resulting source code.
This will make this library useable from other languages by writing simple client wrappers.

For more information proceed to the `project documentation page <http://tigger.publicfields.net>`_.

Tests can be run by installing `Py.Test <http://pytest.org>`_ and running ``py.test`` from the ``test`` folder (run ``py.test --help`` to get the list of options).
