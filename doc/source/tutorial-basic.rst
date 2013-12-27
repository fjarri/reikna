.. _tutorial-basic:

****************
Tutorial: basics
****************

Usage of computations
=====================

All ``reikna`` computation classes are derived from the :py:class:`~reikna.core.Computation` class and therefore share the same API and behavior.
A computation object is an opaque typed function-like object containing all the information necessary to generate GPU kernels that implement some algorithm, along with necessary internal temporary and persistent memory buffers.
Before use it needs to be compiled by calling :py:meth:`~reikna.core.Computation.compile` for a given :py:class:`~reikna.cluda.api.Thread` (thus using its associated device and queue).
This method returns a :py:class:`~reikna.core.computation.ComputationCallable` object which takes GPU arrays and scalar parameters and calls its internal kernels.


Computations and transformations
================================

One often needs to perform some simple processing of the input or output values of a computation.
This can be scaling, splitting complex values into components, padding, and so on.
Some of these operations require additional memory to store intermediate results, and all of them involve additional overhead of calling the kernel, and passing values to and from the device memory.
``Reikna`` porvides an API to define such transformations and attach them to "core" computations, effectively compiling the transformation code into the main kernel(s), thus avoiding all these drawbacks.


Transformation tree
===================

Before talking about transformations themselves, we need to take a closer look at the computation signatures.
Every :py:class:`~reikna.core.Computation` object has a :py:attr:`~reikna.core.Computation.signature` attribute containing ``funcsigs.Signature`` object.
It is the same signature object as can be exctracted from any Python function using ``funcsigs.signature`` function (or ``inspect.signature`` from the standard library for Python >= 3.3).
When the computation object is compiled, the resulting callable will have this exact signature.

The base signature for any computation can be found in its documentation (and, sometimes, can depend on the arguments passed to its constructor --- see, for example, :py:class:`~reikna.algorithms.PureParallel`).
The signature can change if a user connects transformations to some parameter via :py:meth:`~reikna.core.Computation.connect`; in this case the :py:attr:`~reikna.core.Computation.signature` attribute will change accordingly.

All attached transformations form a tree with roots being the base parameters computation has right after creation, and leaves forming the user-visible signature, which the compiled :py:class:`~reikna.core.computation.ComputationCallable` will have.

As an example, let us consider a pure parallel computation object with one output, two inputs and a scalar parameter, which performs the calculation ``out = in1 + in2 + param``:

.. testcode:: transformation_example

    from __future__ import print_function
    import numpy

    from reikna import cluda
    from reikna.cluda import Snippet
    from reikna.core import Transformation, Type, Annotation, Parameter
    from reikna.algorithms import PureParallel
    import reikna.transformations as transformations

    arr_t = Type(numpy.float32, shape=128)
    carr_t = Type(numpy.complex64, shape=128)

    comp = PureParallel(
        [Parameter('out', Annotation(carr_t, 'o')),
        Parameter('in1', Annotation(carr_t, 'i')),
        Parameter('in2', Annotation(carr_t, 'i')),
        Parameter('param', Annotation(numpy.float32))],
        """
        VSIZE_T idx = ${idxs[0]};
        ${out.store_idx}(
            idx, ${in1.load_idx}(idx) + ${in2.load_idx}(idx) + ${param});
        """)

The details of creating the computation itself are not important for this example; they are provided here just for the sake of completeness.
The initial transformation tree of ``comp`` object looks like:

::

       | out   | >>
    >> | in1   |
    >> | in2   |
    >> | param |

Here the insides of ``||`` are the base computation (the one defined by the developer), and ``>>`` denote inputs and outputs provided by the user.
The computation signature is:

.. doctest:: transformation_example

    >>> for param in comp.signature.parameters.values():
    ...     print(param.name + ":" + repr(param.annotation))
    out:Annotation(Type(complex64, shape=(128,), strides=(8,)), role='o')
    in1:Annotation(Type(complex64, shape=(128,), strides=(8,)), role='i')
    in2:Annotation(Type(complex64, shape=(128,), strides=(8,)), role='i')
    param:Annotation(float32)

Now let us attach the transformation to the output which will split it into two halves: ``out1 = out / 2``, ``out2 = out / 2``:

.. testcode:: transformation_example

    tr = transformations.split_complex(comp.parameter.out)
    comp.parameter.out.connect(tr, tr.input, out1=tr.real, out2=tr.imag)

We have used the pre-created transformation here for simplicity; writing custom transformations is described in :ref:`tutorial-advanced-transformation`.

In addition, we want ``in2`` to be scaled before being passed to the main computation.
To achieve this, we connect the scaling transformation to it:

.. testcode:: transformation_example

    tr = transformations.mul_param(comp.parameter.in2, numpy.float32)
    comp.parameter.in2.connect(tr, tr.output, in2_prime=tr.input, param2=tr.param)

The transformation tree now looks like:

::

                         | out   | ----> out1 >>
                         |       |   \-> out2 >>
                      >> | in1   |
    >> in2_prime ------> | in2   |
    >> param2 ----/      |       |
                         | param |

As can be seen, nothing has changed from the base computation's point of view: it still gets the same inputs and outputs to the same array.
But user-supplied parameters (``>>``) have changed, which can be also seen in the value of the :py:attr:`~reikna.core.Computation.signature`:

.. doctest:: transformation_example

    >>> for param in comp.signature.parameters.values():
    ...     print(param.name + ":" + repr(param.annotation))
    out1:Annotation(Type(float32, shape=(128,), strides=(4,)), role='o')
    out2:Annotation(Type(float32, shape=(128,), strides=(4,)), role='o')
    in1:Annotation(Type(complex64, shape=(128,), strides=(8,)), role='i')
    in2_prime:Annotation(Type(complex64, shape=(128,), strides=(8,)), role='i')
    param2:Annotation(float32)
    param:Annotation(float32)

Notice that the order of the final signature is obtained by traversing the transformation tree depth-first, starting from the base parameters.
For more details see the note in the documentation for :py:meth:`~reikna.core.Computation.connect`.

The resulting computation returns the value ``in1 + (in2_prime * param2) + param`` split in half.
In order to run it, we have to compile it first.
When ``prepare_for`` is called, the data types and shapes of the given arguments will be propagated to the roots and used to prepare the original computation.

.. testcode:: transformation_example

    api = cluda.ocl_api()
    thr = api.Thread.create()

    in1_t = comp.parameter.in1
    in2p_t = comp.parameter.in2_prime

    out1 = thr.empty_like(comp.parameter.out1)
    out2 = thr.empty_like(comp.parameter.out2)
    in1 = thr.to_device(numpy.ones(in1_t.shape, in1_t.dtype))
    in2_prime = thr.to_device(numpy.ones(in2p_t.shape, in2p_t.dtype))

    c_comp = comp.compile(thr)
    c_comp(out1, out2, in1, in2_prime, 4, 3)


Transformation restrictions
===========================

There are some limitations of the transformation mechanics:

#. Transformations are purely parallel, that is they cannot use local memory.
   In fact, they are very much like :py:class:`~reikna.algorithms.PureParallel` computations,
   except that the indices they use are defined by the main computation,
   and not set by the GPU driver.
#. External endpoints of the output transformations cannot point to existing nodes in the transformation tree.
   This is the direct consequence of the first limitation --- it would unavoidably create races between memory writes from different branches.
   On the other hand, input transformations can be safely connected to existing nodes, including base nodes (although note that inputs are not cached; so even if you load twice from the same index of the same input node, the global memory will be queried twice).
