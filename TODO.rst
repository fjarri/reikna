0.2.3 (CLUDA API change)
========================

* TODO: write a tutorial on modules
* DECIDE: think of better way of module discovery in render keywords than looking inside AttrDicts. (see reikna.cluda.kernel.process_render_kwds)
* DECIDE: positional arguments for modules?


0.3.0 (Core API change)
========================

* TODO: add comments to ``core.transformation`` and refactor its templates
* TODO: change the misleading name Elementwise to PureParallel (or something)
* TODO: use classes instead of functions transformations

* DECIDE: keyword arguments only?

  * can mark an argument as both input and output
  * easier to construct and return signature
  * easier to handle internally
  * can return dependencies between external arguments with the signature
  * can allow to omit some of the positional arguments during the preparation
    and deduce their shape (i.e., direction in FFT)
  * arguments (and their types/shapes) can be available as attributes of the computation object
  * any disadvantages?

* DECIDE: several methods in the same Computations?

  * FFT, DHT, CBRNG can take advantage of that
  * connect as ``fft.forward.output.connect(...)``
  * which method prepare_for() uses? Or just use some general prepare()?

* TODO: use different classes for different states of Computation

  * ready for setting arglist: ComputationFactory?
  * ready for connects/prepare: ComputationTemplate?
  * ready for calls: Computation

* TODO: take not only CLUDA Thread as a parameter for computation constructor, but also CommandQueue, opencl Context, CUDA Stream and so on.

* TODO: move some of the functionality to the top level of ``reikna`` module?


0.3.1
=====

* TODO: use modules in ``CBRNG``
* DECIDE: add ability to manually override inferred dependencies?
* TODO: add support for arrays with aligned rows (mem_alloc_pitch() in PyCuda).
  This should make non-power-of-2 FFT much faster.
* DECIDE: move all "raw" computations to their own submodule?
* TODO: document _debug usage
* TODO: add a global DEBUG variable that will create all computations in debug mode by default
* TODO: add "dynamic regime"
* TODO: run coverage tests and see if some functionality has to be tested,
  and check existing testcases for redundancy (fft and vsizes in particular)
* TODO: run pylint
* TODO: create "fallback" when if _construct_operations() does not catch OutOfResources,
  it is called again with reduced local size
* TODO: add special optimized kernel for matrix-vector multiplication in MatrixMul.
  Or create specific matrix-vector and vector-vector computations?
* TODO: add ``Thread.fork()`` which creates another Thread with the same context and device but different queue.
  Also, how do we create a ``Thread`` with the same context, but different device?
  Or how do we create and use a ``Thread`` with several devices?


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
* TODO: add a mechanism to select the best local size based on occupancy

Core:

* CHECK: check for errors in load/stores/param usage when connecting transformations?
  Alternatively, return more meaningful errors when accessing load/store/parameter with the wrong number.
* CHECK: check for errors in load/stores/param usage in kernels?
  Need to see what errors look like in this case.
* CHECK: check correctness of types in Computation.__call__() if _debug is on
* CHECK: check that types of arrays passed to prepare_for()/received from _get_base_signature() after creating a basis are supported by GPU (eliminates the need to check it in every computation)
* TODO: remove unnecessary whitespace from the transformation code (generated code will look better)
* TODO: cache results of _construct_operations based on the basis, device_params, argnames and attached transformations

Computations:

* CHECK: need to find a balance between creating more workgroups or making loops inside kernels
  (can be applied in elementwise kernels)
* TODO: add bitonic sort
* TODO: add filter
* TODO: add better block width finder for small matrices in matrixmul
* TODO: add radix-3,5,7 for FFT


1.*
===

CLUDA:

* TODO: add support for rational numbers (based on int2)

Core:

* DECIDE: Some mechanism to merge together two successive Computation calls. Will require an API to tell reikna that certain computations are executed together, plus some way to determine if the computation is local and elementwise (otherwise the connection will require the change of code).


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
Take a look at:

* Copperhead (Python-based DSL for GPGPU)
* CodePy (Python -> AST transformer)
* Clyther (subset of Python -> OpenCL code)
* https://github.com/mdipierro/mdpcl (same)
