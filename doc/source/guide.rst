*****
Guide
*****

Basic usage of computations
===========================

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
--------------------------------

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

The class is described here just for reference, the detailed explanation about writing your own computation classes is given in :ref:`the following sections <guide-contributing>`.
Its initial transformation tree looks like:

(pic with base values out, in1, in2, param)

And its signature is

.. doctest:: transformation_example

    >>> comp.signature_str()
    '(array) out, (array) in1, (array) in2, (scalar) param'

Now let us attach the transformation to the output which will split it into two halves: ``out1 = out / 2``, ``out2 = out / 2``:

.. testcode:: transformation_example

    comp.connect(transformations.split_complex, 'out', ['out1', 'out2'])

We have used the pre-created transformation here for simplicity; writing your own transformations will be described :ref:`later <guide-write-transformations>`.
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


Mako basics
===========

Tigger uses `Mako <http://makotemplates.org>`_ extensively as a templating engine for transformations and computations.
For the purpose of this guide you only need to know several things about the synthax:

* Most of Mako synthax is plain Python, with the set of global variables specified externally by the code doing the template rendering
* ``${expr}`` evaluates Python expression ``expr``, calls ``str()`` on the result and puts it into the text
* a pair of ``<%`` and ``%>`` executes Python code inside, which may introduce some local variables
* a pair of ``<%def name="func(a, b)">`` and ``</%def>`` defines a template function, which actually becomes a Python function which can be called as ``func(a, b)`` from other part of the template and returns a rendered string


Writing a transformation
========================

Some common transformations are already available from :py:mod:`~tigger.transformations` module.
But you can create a custom one if you need to.
Transformations are based on the class :py:class:`~tigger.core.Transformation`.
Its constructor has three major groups of parameters.

First, ``outputs``, ``inputs`` and ``parameters`` are numbers specifying how many arguments of corresponding type the transformation take.

Second, four ``derive_X_from_Y`` options take lambdas that perform type derivation.
This happens when any of the preparation functions is called; therefore the derivation in both directions is required.
In addition, some transformations (like :py:func:`~tigger.transformations.scale_param`) can serve both as input and as output transformations.
Therefore the total of four transformations is required, although two is enough if the transformation is used only for input or only for output.

The format of required lambdas is the following (here ``iN``, ``oN`` and ``pN`` are :py:class:`numpy.dtype` objects):

* ``derive_o_from_is(i1, ..., p1, ...)`` is called when the transformation is connected to the output node, and the derivation from the *root* nodes is required (:py:meth:`~tigger.core.Computation.prepare` was called, or the basis was changed by :py:meth:`~tigger.core.Computation.prepare_for`).
  Returns an iterable ``(o1, ...)``.
* ``derive_is_from_o(o1, ...)`` is called when the transformation is connected to the output node, and the derivation from the *leaf* nodes is required (:py:meth:`~tigger.core.Computation.prepare_for` was called).
  Returns a pair of iterables ``(i1, ...), (p1, ...)``.
* ``derive_i_from_os(o1, ..., p1, ...)`` is called when the transformation is connected to the *input* node, and the derivation from the *leaf* nodes is required (:py:meth:`~tigger.core.Computation.prepare_for` was called).
  Returns an iterable ``(i1, ...)``.
* ``derive_os_from_i(i1, ...)`` is called when the transformation is connected to the *input* node, and the derivation from the *root* nodes is required .
  Returns a pair of iterables ``(o1, ...), (p1, ...)`` (:py:meth:`~tigger.core.Computation.prepare` was called, or the basis was changed by :py:meth:`~tigger.core.Computation.prepare_for`).

The last part of the constructor is a ``code`` parameter.
It is a string with the Mako template which describes the transformation.
Variables ``i1``, ..., ``o1``, ..., ``p1``, ... are available in the template and help specify load and store actions for inputs, outputs and parameters, and also to obtain their datatypes.
Each of these variables has attributes ``dtype`` (contains the :py:class:`numpy.dtype`), ``ctype`` (contains a string with corresponding C type) and either of ``load`` (for inputs) or ``store`` (for outputs).
``${i1.load}`` can be used as a variable, and ``${o1.store}(val)`` is a function that takes one variable.
Also the ``dtypes`` variable is available in the template and gives access :py:mod:`~tigger.cluda.dtypes` module, and ``func`` is a module-like object containing generalizations of arithmetic functions (see :ref:`kernel-toolbox` for details).

For example, for a scaling transformation with one input, one output and one parameter the code may look like:

::

    ${o1.store}(${func.mul(i1.dtype, p1.dtype, out=o1.dtype)}(${i1.load}, ${p1}));

There is a lot of stuff going on in this single line.
First, notice that the input is loaded as ``${i1.load}``, and the parameter as ``${p1}``.
Second, since any of the ``i1`` and ``p1`` can be complex, we had to use the generic multiplication template from the ``func`` quasi-module.
The result is passed to the output by calling ``${o1.store}``.
If the transformation has several outputs, it will have several ``store`` statements.


Writing a computation
=====================

A computation must derive :py:class:`~tigger.core.Computation` class and implement several methods.
As an example, let us implement a computation which calculates ``output = input1 + input2 * param``.

Defining a class:

::

    class TestComputation(Computation):

First, we have to specify :py:meth:`~tigger.core.Computation._get_argnames` which returns argument names for the computation.
The arguments are split into three groups: outputs, inputs and scalar arguments.

::

    def _get_argnames(self):
        return ('output',), ('input1', 'input2'), ('param',)

If you do not implement this method, :py:meth:`~tigger.core.Computation.set_argnames` method will be available to users, and supplied argument names will be passed to other methods discussed below as ``argnames`` parameter.
This is how computations with variable arguments, like :py:class:`~tigger.elementwise.Elementwise` are defined.

Then you need to think about what values will constitute a basis for the computation.
Basis should contain all the information to fully specify kernels, allocations and all other computation details.
In our case, we will force all the variables have the same data type (although it is not necessary).
In addition we will need to add the array size to the basis.
Method :py:meth:`~tigger.core.Computation._get_default_basis` returns a dcitionary with default values for the basis:

::

    def _get_default_basis(self):
