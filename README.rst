==========================================================
Codename Tigger: pure Python library with GPGPU algorithms
==========================================================

Tigger is the attempt to combine all the algorithms I am currently using in my projects into one library.
In particular, PyFFT will be merged into this project too.
My goals are:

* gather a number of basic algorithms into one place, so that they could be documented and covered by tests;
* make this library Python-only (no compilation issues);
* provide identical behaviour for both Cuda and OpenCL (so that it is enough to change a single line of code to switch from one to the other);
* make it easy for other people to add new algorithms.

The project is in the prototype stage now, and everything is subject to change.
It may even disappear completely if, for example, it is decided that it should be joined with Compyte.
