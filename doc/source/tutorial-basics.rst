****************
Tutorial: basics
****************

Usage of computations
=====================

All Tigger computation classes are derived from :py:class:`~tigger.core.Computation` class and therefore share the same API and behavior.
Each computation is parametrized by a dictionary called *basis*, and, sometimes, by names and positions of its arguments (when they can vary, for example, in :py:class:`~tigger.elementwise.Elementwise`).

Before use a computation has to be fully prepared by calling either :py:meth:`~tigger.core.Computation.prepare` or :py:meth:`~tigger.core.Computation.prepare_for`.
The former method one directly reassigns values in the basis:

::

    tr = Transpose(ctx).prepare(dtype=numpy.float64, input_shape=(200, 100))

Here we are preparing transposition object with explicitly set data type and input shape, but with ``axes`` parameter keeping its default value ``None``.
The latter method derives basis from a set of positional arguments and optional keyword arguments, where the arguments are the same as you are going to pass to :py:meth:`~tigger.core.Computation.__call__`:

::

    input = ctx.allocate((100, 200), dtype=numpy.float32)
    output = ctx.allocate((200, 100), dtype=numpy.float32)
    tr = Transpose(ctx).prepare_for(output, input)
    tr(output, input)

Here ``dtype`` and ``input_shape`` were derived from shapes and types of ``output`` and ``input`` values.

Consequently, API of each computation class is fully defined by documenting these two preparation functions. Namely, :py:meth:`~tigger.core.Computation.__call__` has the same positional arguments as :py:meth:`~tigger.core.Computation.prepare_for`, :py:meth:`~tigger.core.Computation.set_basis` has the same arguments as :py:meth:`~tigger.core.Computation.prepare`, and so on.


Computations and transformations
================================

One often needs to perform some simple processing on the input or output values of a computation.
This can be scaling, splitting complex values into components, and so on.
Some of them require additional memory to store intermediate results, and all of them involve additional overhead of calling the kernel, and passing values to and from device memory.
Tigger porvides an API to write such transformations and attach them to "core" computations, effectively compiling the transformation code into the main kernel, thus avoiding all these drawbacks.

Transformation tree
===================

Before talking about transformations themselves, we need to take a closer look at computation signatures.
Positional arguments of any :py:meth:`~tigger.core.Computation.__call__` method are output buffers, input buffers, and scalar arguments, in this order.
All these values are eventually passed to the computation kernel.
Also, all these values have a name, which can be seen in the documentation for the :py:meth:`~tigger.core.Computation.prepare_for` method of the corresponding computation.
These names serve as identifiers for connection points, where user can attach transformations.

All attached transformations form a tree with roots being these base connection points, and leaves forming the signature of the :py:meth:`~tigger.core.Computation.__call__` method visible to the user.
As an example, let us consider an elementwise computation object with one output, two inputs and a scalar parameter, which performs the calculation ``out = in1 + in2 + param``:

.. testcode:: transformation_example

    import numpy
    import tigger.cluda as cluda
    from tigger.core import Transformation
    from tigger.elementwise import specialize_elementwise
    import tigger.transformations as transformations

    api = cluda.ocl_api()
    ctx = api.Context.create()

    TestComputation = specialize_elementwise(
        'out', ['in1', 'in2'], 'param',
        dict(kernel="${out.store}(${in1.load} + ${in2.load} + ${param};"))

    comp = TestComputation(ctx)

The class is described here just for reference, the detailed explanation about writing your own computation classes is given in :ref:`the guide <guide-writing-a-computation>`.
Its initial transformation tree looks like:

(pic with base values out, in1, in2, param)

And its signature is

.. doctest:: transformation_example

    >>> comp.signature_str()
    '(array) out, (array) in1, (array) in2, (scalar) param'

Now let us attach the transformation to the output which will split it into two halves: ``out1 = out / 2``, ``out2 = out / 2``:

.. testcode:: transformation_example

    comp.connect(transformations.split_complex, 'out', ['out1', 'out2'])

We have used the pre-created transformation here for simplicity; writing your own transformations will be described :ref:`later <guide-writing-a-transformation>`.
In addition, we want ``in2`` to be scaled before being passed to the main computation.
To achieve this, we connect the scaling transformation to it:

.. testcode:: transformation_example

    comp.connect(transformations.scale_param, 'in2', ['in2_prime'], ['param2'])

The transformation tree now looks like (blue contour shows the external signature, arrows show the direction of data):

(pic with new tree)

And the signature is:

.. doctest:: transformation_example

    >>> comp.signature_str()
    '(array) out1, (array) out2, (array) in1, (array) in2_prime, (scalar) param, (scalar) param2'

Notice that ``param2`` was moved to the end of the signature.
This was done in order to keep outputs, inputs and scalar parameters separated.
Except for that, the order of the final signature is obtained by traversing the transformation tree depth-first.

The resulting computation returns value ``in1 + (in2_prime * param2) + param`` split in half.
In order to run it, we have to prepare it first.
If :py:meth:`~tigger.core.Computation.prepare` is called, the data types and shapes for each of the value in the tree will be propagated from the roots.
If :py:meth:`~tigger.core.Computation.prepare_for` is called, the data types and shapes will be propagated to the roots and used to prepare the original computation.

::

    comp.prepare_for(out1, out2, in1, in2_prime, param, param2)
    comp(out1, out2, in1, in2_prime, param, param2)


Transformation restrictions
===========================

#. Transformations are strictly elementwise.
   It means that you cannot specify the index to read from or to write to in the transformation code --- it stays the same as the one in the main kernel.
#. Transformations connected to the input nodes must have only one output, and transformations connected to the output nodes must have only one input.
   This restriction is, in fact, enforced by the signature of :py:meth:`~tigger.core.Computation.connect`.
#. External endpoints of the output transformations cannot point to existing nodes in the transformation tree.
   This is the direct consequence of the strict elementwiseness --- it would unavoidably create races between memory writes from different branches.
   On the other hand, input transformations can be safely connected to existing nodes, including base nodes.
