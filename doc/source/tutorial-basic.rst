.. _tutorial-basic:

****************
Tutorial: basics
****************

Usage of computations
=====================

All ``reikna`` computation classes are derived from the :py:class:`~reikna.core.Computation` class and therefore share the same API and behavior.
Each computation is parametrized by a dictionary called **basis** (which is hidden from the user), and, sometimes, by the names and positions of its arguments (when they can vary, for example, in :py:class:`~reikna.elementwise.Elementwise`).

Before use a computation has to be fully prepared by means of calling :py:meth:`~reikna.core.Computation.prepare_for`.
This method derives the basis from a set of positional arguments and optional keyword arguments.
The positional arguments should be either the same arrays and scalars you are going to pass to the computation call (which means the same shapes and data types), or their replacements in the form of :py:class:`~reikna.core.ArrayValue` and :py:class:`~reikna.core.ScalarValue` objects:

::

    input = thr.array((100, 200), dtype=numpy.float32)
    output = thr.array((200, 100), dtype=numpy.float32)
    tr = Transpose(thr).prepare_for(output, input)
    tr(output, input)

Consequently, API of each computation class is fully defined by the documentation for its ``prepare_for`` method.
In particular, ``__call__`` has the same positional arguments as ``prepare_for``, and base computation argument names (used to attach transformations) are the names of these positional arguments.


Computations and transformations
================================

One often needs to perform some simple processing of the input or output values of a computation.
This can be scaling, splitting complex values into components, and so on.
Some of these operations require additional memory to store intermediate results, and all of them involve additional overhead of calling the kernel, and passing values to and from the device memory.
``Reikna`` porvides an API to define such transformations and attach them to "core" computations, effectively compiling the transformation code into the main kernel, thus avoiding all these drawbacks.

Transformation tree
===================

Before talking about transformations themselves, we need to take a closer look at the computation signatures.
Positional arguments of any ``__call__`` method of a class derived from :py:meth:`~reikna.core.Computation` are output arrays, input arrays, and scalar arguments, in this order.
All these values are eventually passed to the computation kernel.

All the positional arguments have an identifier which is unique for the given computation object.
Identifiers for the base computation (without any connected transformation) are, by convention, the names of the positional arguments to ``prepare_for`` for the computation.
These identifiers serve as connection points, where the user can attach transformations.

All attached transformations form a tree with roots being these base connection points, and leaves forming defining the positional arguments to ``prepare_for`` and ``__call__`` methods visible to the user.
As an example, let us consider an elementwise computation object with one output, two inputs and a scalar parameter, which performs the calculation ``out = in1 + in2 + param``:

.. testcode:: transformation_example

    import numpy
    from reikna import cluda
    from reikna.cluda import Module
    from reikna.core import Transformation
    from reikna.elementwise import specialize_elementwise
    import reikna.transformations as transformations

    api = cluda.ocl_api()
    thr = api.Thread.create()

    code = lambda out_dtype, in1_dtype, in2_dtype, param_dtype: Module.create(
        lambda out, in1, in2, param:
            """
            ${out.store}(idx, ${in1.load}(idx) + ${in2.load}(idx) + ${param});
            """,
        snippet=True)

    TestComputation = specialize_elementwise(
        'out', ['in1', 'in2'], 'param', code)

    comp = TestComputation(thr)

The details of creating the ``TestComputation`` class are not important for this example; they are provided here just for the sake of completeness.
The initial transformation tree of ``comp`` object looks like:

::

       | out   | >>
    >> | in1   |
    >> | in2   |
    >> | param |

Here ``||`` denote the base computation (the one defined by the developer), and ``>>`` denote inputs and outputs specified by the user.
The computation signature is:

.. doctest:: transformation_example

    >>> comp.signature_str()
    '(array) out, (array) in1, (array) in2, (scalar) param'

Now let us attach the transformation to the output which will split it into two halves: ``out1 = out / 2``, ``out2 = out / 2``:

.. testcode:: transformation_example

    comp.connect(transformations.split_complex(), 'out', ['out1', 'out2'])

We have used the pre-created transformation here for simplicity; writing custom transformations is described in :ref:`tutorial-advanced-transformation`.

In addition, we want ``in2`` to be scaled before being passed to the main computation.
To achieve this, we connect the scaling transformation to it:

.. testcode:: transformation_example

    comp.connect(transformations.scale_param(), 'in2', ['in2_prime'], ['param2'])

The transformation tree now looks like:

::

                         | out   | ----> out1 >>
                         |       |   \-> out2 >>
                      >> | in1   |
    >> in2_prime ------> | in2   |
                   /  >> | param |
    >> param2 ----/

As can be seen, nothing has changed from the base computation's point of view: it still gets the same inputs and outputs to the same array.
But user-supplied parameters (``>>``) have changed, which can be also seen in the result of the :py:meth:`~reikna.core.Computation.signature_str`:

.. doctest:: transformation_example

    >>> comp.signature_str()
    '(array) out1, (array) out2, (array) in1, (array) in2_prime, (scalar) param, (scalar) param2'

Notice that ``param2`` was moved to the end of the signature.
This was done in order to keep outputs, inputs and scalar parameters grouped.
Except for that, the order of the final signature is obtained by traversing the transformation tree depth-first.

The resulting computation returns the value ``in1 + (in2_prime * param2) + param`` split in half.
In order to run it, we have to prepare it first.
When ``prepare_for`` is called, the data types and shapes of the given arguments will be propagated to the roots and used to prepare the original computation.

.. testcode:: transformation_example

    N = 128
    out1 = thr.array(N, numpy.float32)
    out2 = thr.array(N, numpy.float32)
    in1 = thr.to_device(numpy.ones(N, numpy.float32))
    in2_prime = thr.to_device(numpy.ones(N, numpy.float32))
    param = 3
    param2 = 4
    comp.prepare_for(out1, out2, in1, in2_prime, param, param2)
    comp(out1, out2, in1, in2_prime, param, param2)


Transformation restrictions
===========================

There are some limitations of the transformation mechanics:

#. Transformations are strictly elementwise.
   It means that you cannot specify the index to read from or to write to in the transformation code --- it stays the same as the one used to read the value in the main kernel.
#. Transformations connected to the input nodes must have only one output, and transformations connected to the output nodes must have only one input.
   This restriction is, in fact, enforced by the signature of :py:meth:`~reikna.core.Computation.connect`.
#. External endpoints of the output transformations cannot point to existing nodes in the transformation tree.
   This is the direct consequence of the strict elementwiseness --- it would unavoidably create races between memory writes from different branches.
   On the other hand, input transformations can be safely connected to existing nodes, including base nodes.
