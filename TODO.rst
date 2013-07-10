0.3.0 (Core API change)
========================

To merge the new API branch:

* DOCS: add docstrings for main core classes and check that doctests pass
* FIX: update existing Computations and Transformations to a new API

After merging:

* DOC: write correlations tutorial
* DOC: since ``store_same`` is translated to ``return`` in the input transformations, it is necessary to emphasize in docs that it should be the last instruction in the code.
  Or, alternatively, replace it with something like ``ctype _temp = ...; <rest of the code> return _temp;``?
* TEST (core): Need to test connections to ``'io'`` parameters, and also the behavior of transformations with ``'io'`` parameters (including their use in ``PureParallel``).
  I expect there will be quite a few bugs.
* ?API (computations): move some of the functionality to the top level of ``reikna`` module?
* ?API (core): pass ``kernel_definition`` as a positional argument to a kernel template def?
* ?API (core): when we are connecting a transformation to an existing scalar parameter, during the leaf signature building it moves from its place, which may seem a bit surprising. Should we attempt to keep it at the same place? Like, keep all parameters with default values at the end (thus breaking the "depth first" order, of course)?
* ?API (core): misleading use of the word "dependencies" on CLUDA level and on Computation level: in the latter they are rather "decorrelations" and can exist, say, between two input nodes.
  Perhaps it is worth adding a special section with their strict definition to the docs
* FEATURE (core): take not only CLUDA Thread as a parameter for computation ``compile``, but also CommandQueue, opencl Context, CUDA Stream and so on.
* ?FIX (core): check if Signature.bind() is too slow in the kernel call; perhaps we will have to rewrite it taking into account restrictions to Parameter types we have.
* FIX (core): clean up core code, add comments
* FIX (core): rewrite all internal classes using ``collections.namedtuple`` factory to force their immutability.
* ?FIX (core): need to make available only those ``ComputationArgument`` objects that are actually usable: root ones for the plan creator, and all for the user connecting transformations.
But techically the plan creator does not know anything about connections anyway, so it is not that important.
* FIX (core): When we connect a transformation, difference in strides between arrays in the connection can be ignored (and probably the transformation's signature changed too; at least we need to decide which strides to use in the exposed node)
* FIX (cluda): rewrite vsizes to just use a 1D global size and get any-D virtual sizes through modular division (shouldn't be that slow, but need to test; or maybe just fall back to modular division if the requested dimensionality is too big).
* ?FIX (computations): PureParallel can be either rewritten using stub kernel and Transformation (to use load/store_combined_idx) (downside: order of parameters messes up in this case), or using the new any-D static kernels from CLUDA (if the above fix is implemented).
* FEATURE (computations): processing several indices per thread in PureParallel may result in a performance boost, need to check that.

0.3.1
=====

* FIX (computations): use modules in ``CBRNG``
* FIX (cluda): when ``None`` is passed as a local size for a static kernel, and the global size is small, it sets large values for local size (e.g. for gs=13 it sets ls=480, gs=480).
  It's not critical, just confusing; large global sizes seem to have much less unused threads.
  Also, in general, cluda/vsize code is a mess.
* ?API (computations): move all "raw" computations to their own submodule?
* TESTS: run coverage tests and see if some functionality has to be tested,
  and check existing testcases for redundancy (fft and vsizes in particular)
* TESTS: run pylint
* FEATURE (core): create "fallback" when if _build_plan() does not catch OutOfResources,
  it is called again with reduced local size
* FEATURE (computations): add special optimized kernel for matrix-vector multiplication in MatrixMul.
  Or create specific matrix-vector and vector-vector computations?
* FEATURE (CLUDA): add ``Thread.fork()`` which creates another Thread with the same context and device but different queue.
  Also, how do we create a ``Thread`` with the same context, but different device?
  Or how do we create and use a ``Thread`` with several devices?
* FEATURE (computations): reduction with multiple predicates on a single (or multiple too?) array.
  Basically, the first stage has to be modified to store results in several arrays and then several separate reductions can be performed.
* FEATURE (CLUDA, core): implement custom structures as types (will also require updating the strides-to-flat-index algorithm)
* ?API (CLUDA, core): do something about the inconsistency of array shapes (row-major) and global sizes (column-major). Special get_id() functions maybe?
* ?FEATURE (core): add ``load_flat``/``store_flat`` to argobjects?
* ?FEATURE (core): when passing scalar to plan.kernel_call(), there's no typecheck (and the only check that happens is during execution). Need to somehow query the kernel about type of its parameters.


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
  (can be applied in pure parallel kernels)
* TODO: add bitonic sort
* TODO: add filter
* TODO: add better block width finder for small matrices in matrixmul
* TODO: add radix-3,5,7 for FFT


1.*
===

CLUDA:

* TODO: add support for rational numbers (based on int2)

Core:

* DECIDE: Some mechanism to merge together two successive Computation calls. Will require an API to tell reikna that certain computations are executed together, plus some way to determine if the computation is local and pure parallel (otherwise the connection will require the change of code).

* DECIDE: Some mechanism to detect when two transformations are reading from the same node at the same index, and only read the global memory once. This can be done by storing node results in kernel-global variables instead of chaining functions like it's done now. The problem is that we have to be able to distinguish between several loads from the same node at different indices.

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
