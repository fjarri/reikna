0.2.0
=====

* TODO: add full support of numpy.fft.fftn interface to FFT
* TODO: FFT: remove hardcode of max_lmem_fft_size
* TODO: FFT: process cases of problem_size == 1 properly (we still have to run transformations)
* TODO: add memory packing functionality to OperationRecorder
* TODO: cool feature: process the list and remove unnecessary allocations, replacing them by creating views
* TODO: new operation.add_view() function:
  1) If applied to an already created temporary array, just creates its view.
  2) If applied to an output array, check the transformations attached to it.
     If they finish in the single array of the same size or larger, create a view.
     Otherwise create an allocation.
  3) If applied to an output array, and the allocation was already created, create a view of it.
  For testing purposes, 2) and 3) are not necessary - just create a new allocation.
* TODO: add Python 3 supports and run full set of tests


0.3.0
=====

* TODO: move part of core.transformation to a template
* TODO: add custom render keywords for transformations (will help e.g. in tigger.transformations)
* TODO: Flatten kernel list before execution, and assign argument numbers
* TODO: create some elementwise computations derived from Elementwise
* TODO: document _debug usage
* TODO: add "dynamic regime"
* TODO: add "raises" sections to Computation/OperationRecorder methods
* TODO: create "fallback" when if _construct_operations() does not catch OutOfResources,
  it is called again with reduced local size
* TODO: run coverage tests and see if some functionality has to be tested,
  and check existing testcases for redundancy (fft and vsizes in particular)
* TODO: run pylint


1.0.0 (production-quality version... hopefully)
===============================================

Website/documentation:

* TODO: extend starting page (link to issue tracker, quick links to guides, list of algorithms, quick example)

CLUDA:

* DECIDE: does the forceful enabling of double precision in OpenCL somehow change the performance for single precision?
* DECIDE: Is there a way to get number of shared memory banks and warp size from AMD device?
* DECIDE: what are we going to do with OpenCL platforms that do not support intra-block interaction?
  (for example, Apple's implementation)
* DECIDE: make dtypes.result_type() and dtypes.min_scalar_type() depend on device?
* DECIDE: change type of id()/size() functions to size_t in case of CUDA?
* TODO: find a way to get ``min_mem_coalesce_width`` for OpenCL

Core:

* DECIDE: drop strict positioning of outputs-inputs-params and just set argument types?
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
* TODO: optimize allocation calls throughout the context
* TODO: cache results of _construct_operations based on the basis, device_params, argnames and attached transformations

Computations:

* CHECK: need to find a balance between creating more workgroups or making loops inside kernels
  See, for example, matrximul (batch is processed using loop)
* TODO: add DHT
* TODO: add random number generation (MD5 and DCMT seem to be the best candidates)
* TODO: add bitonic sort
* TODO: add filter


1.*
===

CLUDA:

* TODO: add support for rational numbers (based on int2)


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
