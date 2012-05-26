===============
Codename Tigger
===============

Tigger is an attempt to combine all the algorithms I am currently using in my projects into one library.
In particular, PyFFT will be merged into this project too.
My goals are:

* gather a number of basic algorithms into one place, so that they could be documented and covered by tests;
* make this library Python-only (no compilation issues);
* provide identical behaviour for both Cuda and OpenCL (so that it is enough to change a single line of code to switch from one to the other);
* make it easy for other people to add new algorithms.

Long-term goal: library that by request (perhaps, from other languages) returns kernels and call signatures for algorithms, using Python as templating engine.

The project is in the prototype stage now, and everything is subject to change.
It may even disappear completely if, for example, it is decided that it should be joined with Compyte.

---------------
Release history
---------------

v0.0.1 (planned)
----------------

Main tasks:

* Add the following algorithms: matrix multiplication, transposition, 3D permutation, FFT, DHT, reduction.
* Add pre- and post-processing for algorithms.
* Add basic documentation.

Additional tasks:

* Add some global DEBUG variable, it will help with testing.
* Improve Env creation: they have to be able to "connect" to existing contexts/queues. It seems to be, in fact, the main usage scenario.