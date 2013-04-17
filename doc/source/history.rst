***************
Release history
***************


0.2.2 (current development version)
===================================

* ADDED: ability to use custom argument names in transformations.

* ADDED: multi-argument :py:meth:`~reikna.cluda.kernel.FuncCollector.mul`.

* ADDED: counter-based random number generator :py:class:`~reikna.cbrng.CBRNG`.

* ADDED: :py:class:`~reikna.elementwise.Elementwise` now supports argument dependencies.

* ADDED: Module support in CLUDA; see :ref:`tutorial-modules` for details.

* REMOVED: ``cluda.kernel.render_template_source``; :py:func:`~reikna.cluda.kernel.render_template` should be used instead

* FIXED: removed boilerplate code from :py:class:`~reikna.cluda.kernel.FuncCollector`,
  and fixed some inconsistencies in the function signatures.

* FIXED: some OpenCL compilation warnings caused by an incorrect version querying macro.

* FIXED: bug with incorrect processing of scalar global size in static kernels.


0.2.1 (8 Mar 2013)
==================

* FIXED: function names for kernel ``polar()``, ``exp()`` and ``conj()``.

* FIXED: added forgotten kernel ``norm()`` handler.

* FIXED: bug in ``Py.Test`` testcase execution hook which caused every test to run twice.

* FIXED: bug in nested computation processing for computation with more than one kernel.

* FIXED: added dependencies between :py:class:`~reikna.matrixmul.MatrixMul` kernel arguments.

* FIXED: taking into account dependencies between input and output arrays as well as the ones
  between internal allocations --- necessary for nested computations.

* ADDED: discrete harmonic transform :py:class:`~reikna.dht.DHT`
  (calculated using Gauss-Hermite quadrature).


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
