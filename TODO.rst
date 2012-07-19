0.1.0 (minimal useable version)
===============================

Website/documentation:

* TODO: separate dev/ and stable/ versions of documentation
* TODO: extend starting page (link to issue tracker, quick links to guides, list of algorithms, quick example)

CLUDA:

* TODO: is CLUDA even a good name?
* TODO: implement proper dtype-ctype correspondences (like in compyte)
* TODO: supports_dtype() method should be extended to check for availability of other types (i.e. no floats/ints with size lesser than 4 are available, although they exist in numpy)
* FIXME: add a warning in docs that CPU array used in async operations must be pagelocked
* TODO: get number of shared memory banks from device
* TODO: get warp size from device
* TODO: does the forceful enabling of double precision in OpenCL somehow change the performance for single precision?
* TODO: what are we going to do with OpenCL platforms that do not support intra-block interaction?
  (for example, Apple's implementation)
* TODO: add functions for devices/platforms discovery, and generalized analogue of make_some_context (with interactive option, perhaps)

Core:

* TODO: check for errors in load/stores/param usage when connecting transformations?
  Alternatively, return more meaningful errors when accessing load/store/parameter with the wrong number.
* TODO: check for errors in load/stores/param usage in kernels?
  Need to see what errors look like in this case.
* TODO: check correctness of types in Computation.__call__() if _debug is on
* TODO: add some more "built-in" variables to kernel rendering? Numpy, helpers?
* TODO: profile Computation.__call__() and see if it takes too long, and if the algorithm of assignment args to endpoints should be improved.
* TODO: check that the transformation only has one load/store in the place where connection occurs
* TODO: remove unnecessary whitespace from the transformation code (generated code will look better)
* TODO: add a global DEBUG variable that will create all computations in debug mode by default
* TODO: how to handle external calls, like Transpose in Reduce?
  (Solution: we request the same execution list from Transpose, set argument names - should be a method for that - and incorporate it into our own list)
* TODO: add support to Allocate as the operation, add internally allocated arrays to arg_dict
* TODO: cool feature: process the list and remove unnecessary allocations, replacing them by creating views
* TODO: add usual transformations and derivation functions for convenience
* TODO: check that types of arrays passed to prepare_for()/received from _get_base_signature() after creating a basis are supported by GPU (eliminates the need to check it in every computation)
* TODO: prefix variables from signature with something to avoid clashes in code
* TODO: if None is passed to prepare_for(), transform it to empty ArrayValue/ScalarValue (_construct_basis may work even if some arrays are undefined; for example, result array can be derived from arguments)

Computations:

* TODO: add elementwise computation
* TODO: add reduction
* TODO: add FFT (and mark pyfft as deprecated)
* TODO: add DHT
* TODO: add 3D permutations
* TODO: add random number generation (MD5 and DCMT seem to be the best candidates)
* TODO: add bitonic sort
* TODO: create policy for wrapping raw computations into more convenient classes
* TODO: create policy for providing pre-made computations like sin()/cos()


0.0.1 (prototype version)
=========================

Core:

* FIXME: add endpoint validity check to Computation.connect().
  We can use some of the existing nodes as endpoints, they just need to be not base ones
* FIXME: fix arg assignment in Computation.__call__().
  Basically, for each kernel call we need to calculate leaf signature (taking into account any internal parameters), and assign args according to this signature.
* FIXME: normalize signature of Transformation.leaf_signature().
  The problem is that if base_name is given, returned signature does not include scalars.
  This should be made more clear.
* FIXME: is it good to create new Scalar/ArrayValues in Transformation._clear_values?
* FIXME: instead of passing numpy in Transformation.transformations_for() to give access to datatypes, create a dict with only those types that can be used on GPU and pass it instead.
* TODO: sanity checks in TransformationTree.__init__()? (repeating names, correct identifier format)
* TODO: build_arglist() and signature_macro() in transformation.py are almost identical
* TODO: KernelCall interface looks messy and non-intuitive.

Computations:

* TODO: Do we need to return array shape in _get_base_signature()?

Tests:

* TODO: check computation signatures for correctness (when I decide on the format)


Computation provider (long-term goal)
=====================================

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
