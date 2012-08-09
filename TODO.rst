0.1.0 (minimal useable version)
===============================

Core:

* DECIDE: how to handle external calls, like Transpose in Reduce?
  (Solution: we request the same execution list from Transpose, set argument names - should be a method for that - and incorporate it into our own list)
* DECIDE: there are different kinds of scalar arguments:
  1) Usual ones, that get passed to the kernel. Their values do not affect basis or kernel set.
     Example: scaling coefficient
  2) Operational modes that do not change basis, but can affect kernel set (this can always be replaced by passing them to kernel and let it handle the situation, but it may be slower).
     Example: 'inverse' in FFT
  3) Those that affect the basis and/or the kernel set (and cannot be reduced to cases 1 and 2 without noticeably affecting the performance).
     Example: ???
  This is connected to the problem of "umbrella" classes that provide more convenient interface for raw computations.
  Perhaps it should be their problem, and for raw computations we can assume that all scalar arguments are type 1.
  Also type 3 parameters can be passed as keywords to prepare_for()/__call__()

Computations, first priority:

* DECIDE: create policy for wrapping raw computations into more convenient classes
* DECIDE: create policy for providing pre-made computations like sin()/cos()
* TODO: add sparse reduction
  (wrapper + memory allocations + call to other computation (transposition))


1.0.0 (production-quality version... hopefully)
===============================================

Website/documentation:

* TODO: extend starting page (link to issue tracker, quick links to guides, list of algorithms, quick example)

CLUDA:

* DECIDE: does the forceful enabling of double precision in OpenCL somehow change the performance for single precision?
* DECIDE: Is there a way to get number of shared memory banks and warp size from AMD device?
* DECIDE: what are we going to do with OpenCL platforms that do not support intra-block interaction?
  (for example, Apple's implementation)
* DECIDE: which terminology to perefer, CUDA or OpenCL?
* DECIDE: make dtypes.result_type() and dtypes.min_scalar_type() depend on device?
* DECIDE: OpenCL supports global sizes which are not multiple to warp size.
  This can be used to increase performance (not much though).
  CUDA also supports block sizes which are not multiples of warp size, but since there is no analogue for the global size, arbitrary number of threads are still not achievable.
* TODO: implement better dynamic local size calculation (if local_size=None).

Core:

* CHECK: check for errors in load/stores/param usage when connecting transformations?
  Alternatively, return more meaningful errors when accessing load/store/parameter with the wrong number.
* CHECK: check for errors in load/stores/param usage in kernels?
  Need to see what errors look like in this case.
* CHECK: check correctness of types in Computation.__call__() if _debug is on
* CHECK: check that types of arrays passed to prepare_for()/received from _get_base_signature() after creating a basis are supported by GPU (eliminates the need to check it in every computation)
* TODO: remove unnecessary whitespace from the transformation code (generated code will look better)
* TODO: add a global DEBUG variable that will create all computations in debug mode by default
* TODO: add usual transformations and derivation functions for convenience
* TODO: take not only CLUDA context as a parameter for computation constructor, but also CommandQueue, opencl context, cuda stream and so on.
* DECIDE: profile Computation.__call__() and see if it takes too long, and if the algorithm of assignment args to endpoints should be improved.

Computations:

* CHECK: check if matrixmul with flat blocks is slower than the one with 2D blocks
* TODO: add FFT (and mark pyfft as deprecated)
* TODO: add DHT
* TODO: add 3D permutations
* TODO: add random number generation (MD5 and DCMT seem to be the best candidates)
* TODO: add bitonic sort
* TODO: add filter


1.*
===

Core:

* TODO: cool feature: process the list and remove unnecessary allocations, replacing them by creating views


2.*
===

Computation provider
--------------------

Library that by request (perhaps, from other languages) returns kernels and call signatures for algorithms, using Python as a templating engine.
Namely, it returns:

1. A list of kernels to be executed in turn.
2. Signatures for each of the kernels (with named parameters and their types).
3. Call parameters for each of the kernels (grid, block, shared memory).
4. List of memory blocks to allocate and their names (which will be used to pass them to kernels when necessary according to signatures).

Problems:

1. More involved algorithms cannot be passed between languages this way (the ones that requires synchronization in the middle, like adaptive-step ODE solver, for example).
2. Need to somehow pass device/context to this library from the caller. The kernels will have to be compiled in order to find out the register usage.
3. How to pass type derivation lambdas? Possible solution: limit derivations to <same_as>(x), <definite_type>, <complex_for>(x), <real_for>(x) and create some micro-DSL for passing these as strings.

Transformation DSL
------------------

Currently transformation code is quite difficult to read and write.
Perhaps some DSL can be devised to make it easier?
Even better, if that DSL could be applied to kernels too.
Take a look at Copperhead (Python-based DSL for GPGPU) and CodePy (Python -> AST transformer)
