***************
Release history
***************

0.3.0 (current development version)
===================================

* FIXED: function names for kernel ``polar()``, ``exp()`` and ``conj()``.
* FIXED: added forgotten kernel ``norm()`` handler.
* FIXED: bug in ``Py.Test`` testcase execution hook which caused every test to run twice.
* FIXED: bug in nested computation processing for computation with more than one kernel.
* FIXED: added dependencies between MatrixMul kernel arguments


0.2.0 (3 Mar 2013)
==================

* Added FFT computation (slightly optimized PyFFT version + Bluestein's algorithm for non-power-of-2 FFT sizes)
* Added Python 3 compatibility
* Added Context-global automatic memory packing
* Added polar(), conj() and exp() functions to kernel toolbox
* Changed name because of the clash with `another Tigger <http://www.astron.nl/meqwiki/Tigger>`_.


0.1.0 (12 Sep 2012)
===================

* Lots of changes in the API
* Added elementwise, reduction and transposition computations
* Extended API reference and added topical guides


0.0.1 (22 Jul 2012)
===================

* Created basic core for computations and transformations
* Added matrix multiplication computation
* Created basic documentation
