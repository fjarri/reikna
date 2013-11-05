0.3.7
=====

* FIX (test): test collection runs extremely slowly, especially for Intel device.
* ?FIX (test): FFT tests run with fast math and complex64, and this is too imprecise on the Intel device, which causes them to fail.
* ?FIX (core): perhaps we should memoize parametrized modules too: for example, FFT produces dozens of modules for load and store (because it calls them in a loop).
* FEATURE (CLUDA, core): implement custom structures as types (will also require updating the strides-to-flat-index algorithm)
* FEATURE (computations): use dtypes for custom structures to pass a counter in CBRNG if the sampler is deterministic.
* ?API (computations): can we improve how Predicates for Reduce are defined?
* FEATURE (computations): reduction with multiple predicates on a single (or multiple too?) array.
  Basically, the first stage has to be modified to store results in several arrays and then several separate reductions can be performed.
* FEATURE (computations): allow non-sequential axes in Reduce


0.3.8
=====

* ?FEATURE (core): add ``load_flat``/``store_flat`` to argobjects?
  Basically it's just a synonym for ``load_combined(len(arg.shape))``.
* TEST (computations): add some performance tests for CBRNG
* API (core, computations): use ``arr_like`` instead of ``arr``/``arr_t`` in places where array-like argument is needed.
* ?API (core): make ``device_params`` an attribute of plan or plan factory?
* ?API (cluda): make dtypes.result_type() and dtypes.min_scalar_type() depend on device?
* FEATURE (core): take not only CLUDA Thread as a parameter for computation ``compile``, but also CommandQueue, opencl Context, CUDA Stream and so on.
* FEATURE (core): create "fallback" when if _build_plan() does not catch OutOfResources,
  it is called again with reduced local size
* FEATURE (CLUDA): add ``Thread.fork()`` which creates another Thread with the same context and device but different queue.
  Also, how do we create a ``Thread`` with the same context, but different device?
  Or how do we create and use a ``Thread`` with several devices?

* FIX (core): When we connect a transformation, difference in strides between arrays in the connection can be ignored (and probably the transformation's signature changed too; at least we need to decide which strides to use in the exposed node).
  Proposal: leave it as is; make existing transformations "propagate" strides to results; and create a special transformation that only changes strides (or make it a parameter to the identity one).
  Currently strides are not supported by PyCUDA or PyOpenCL, so this will wait.
  Idea: strides can be passes to compile() (in form of actual arrays, as a dictionary).


1.0.0 (production-quality version... hopefully)
===============================================

* ?FIX (cluda): Is there a way to get number of shared memory banks and warp size from AMD device?
* ?FIX (cluda): find a way to get ``min_mem_coalesce_width`` for OpenCL
* ?FIX (cluda): what are we going to do with OpenCL platforms that do not support intra-block interaction?
  (for example, Apple's implementation)

* FEATURE (cluda): add a mechanism to select the best local size based on occupancy
* ?API (computations): move some of the functionality to the top level of ``reikna`` module?
* ?FEATURE (core): add ability to connect several transformation parameters to one node.
  Currently it is impossible because of the chosen interface (kwds do not allow repettitions).
  This can be actually still achieved by connecting additional identity transformations.
* FEATURE (docs): extend starting page (link to issue tracker, quick links to guides, list of algorithms, quick example)

* ?FEATURE (core): check for errors in load/stores/param usage when connecting transformations?
  Alternatively, return more meaningful errors when accessing load/store/parameter with the wrong number.
* ?FEATURE (core): check for errors in load/stores/param usage in kernels?
  Need to see what errors look like in this case.
* FEATURE (core): check correctness of types in Computation.__call__() if _debug is on
* ?FEATURE (core): check that types of arrays in the computation signature are supported by GPU (eliminates the need to check it in every computation)
* FEATURE (core): add group identifier to temporary allocations, with the guarantee that the allocations with different groups are not packed.
  This may be used to ensure that buffers to be used on different devices are not packed,
  which may be bad since OpenCL tries to preemptively move allocations from device to device.
  It'll help when Threads start to support several devices.

* FEATURE (computations): add matrix-vector and vector-vector multiplication (the latter can probably be implemented just as a specialized ``Reduce``)
* FEATURE (computations): add better block width finder for small matrices in matrixmul
* FEATURE (computations): add scan
* FEATURE (computations): add bitonic sort
* FEATURE (computations): add filter
* FEATURE (computations): add radix-3,5,7 for FFT
* FEATURE (computations): commonly required linalg functions: diagonalisation, inversion, decomposition, determinant of matrices


1.*
===

* ?FEATURE (cluda): add support for rational numbers (based on int2)
* ?FEATURE (core): Some mechanism to merge together two successive Computation calls. Will require an API to tell reikna that certain computations are executed together, plus some way to determine if the computation is local and pure parallel (otherwise the connection will require the change of code).
* ?FEATURE (core): Some mechanism to detect when two transformations are reading from the same node at the same index, and only read the global memory once. This can be done by storing node results in kernel-global variables instead of chaining functions like it's done now. The problem is that we have to be able to distinguish between several loads from the same node at different indices.

2.*
===


Correlations
------------

It is possible to define for any kernel and transformation which pairs of arrays are accessed in a correlated manner, i.e. something like:

\begin{definition}
Data-independent computation (DIC) is a function $F :: ThreadId -> [(MemId, Operation, Index)]$,
where $ThreadId = Int$, $MemId = Int$, $Index = Int$, $Operation = Input | Output$.
\end{definition}

\begin{definition}
DIC is said to have a decorrelation for buffers $m, n \in MemId$ and block size $b$, if
$\exists t_1, t_2 \in ThreadID, i \in Index |
    block(t_1) \ne block(t_2),
    (m, Input or Output, i) \in F(t_1) and (n, Output, i) \in F(t_2)$.
\end{definition}

\begin{theorem}
If, and only if a DIC has a dependency for buffers $m, n$,
then there exists an index $i$ such that
the order of operations accessing it in buffers $m, n$ is undefined,
and at least one of these operations is $Output$.
\end{theorem}

\begin{definition}
DIC is said to have a writing inconsistency for buffers $m, n$, if
$\exists i \in Index, t1, t2 \in ThreadId |
    (m, Output, i) \in F(t) and (n, Output, i) \in F(t)$.
In other words, it does not rewrite the data.
\end{definition}

Simply put, if input and output are correlated, one can supply the same array for both parameters.
Then, when transformations are connected to kernels, we can propagate correlations (i.e. if A and B are correlated, and transformation B->B' is correlated, then A->B' are correlated) and derive correlations for the resulting kernel.
This is the correlation of access, and only depends on array shapes.

In practice there are all sorts of problems:

* correlation does not allow inplace operation if two arrays have different strides
* one needs to formally justify the propagation through attached transformation
* ... including cases when, say, output transformation reads from another array
* ... or if an array is padded and then unpadded - does the correlation still work? does it work for other arrays involved in this transformation?
* ... does it depend on the order and type of access (read/write)?
* how is end user supposed to take advantage of this knowledge?
  It is doubtful that a user will call some methods of the computation to check whether he can use it inplace; he will rather look into documentation.
  Variant: it may be used for error checking; i.e. to test that same array was not passed to decorrelated parameters.
* we cannot use it to pack temporary arrays, because even identically typed arrays are not guaranteed to start at the same physical memory, therefore "inplace" is meaningless for them

So for now I'm ditching this feature.
Temporary memory is still packed, but only taking into account its appearance in kernel arguments.


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
