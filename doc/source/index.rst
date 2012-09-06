*************************************
Tigger, the pure Python GPGPU library
*************************************

|documentation_type_subtitle|

Tigger is a library containing various GPU algorithms.
The main design goals are:

* separation of computation cores (matrix multiplication, random numbers generation etc) from simple transformations on their input and output values (scaling, typecast etc);
* separation of the preparation and execution stage, maximizing the performance of the execution stage at the expense of the preparation stage (in other words, aiming at large simulations)
* partial abstraction from Cuda/OpenCL

Tigger is hosted `on GitHub <http://github.com/Manticore/tigger>`_, issues should be filed there too.
It is not registered in PyPi at the moment.

********
Contents
********

.. toctree::
   :maxdepth: 2

   introduction
   guide
   api/index
   history

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
