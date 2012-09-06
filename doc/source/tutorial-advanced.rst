*************************
Tutorial: advanced topics
*************************

Mako basics
===========

Tigger uses `Mako <http://makotemplates.org>`_ extensively as a templating engine for transformations and computations.
For the purpose of this guide you only need to know several things about the synthax:

* Most of Mako synthax is plain Python, with the set of global variables specified externally by the code doing the template rendering
* ``${expr}`` evaluates Python expression ``expr``, calls ``str()`` on the result and puts it into the text
* a pair of ``<%`` and ``%>`` executes Python code inside, which may introduce some local variables
* a pair of ``<%def name="func(a, b)">`` and ``</%def>`` defines a template function, which actually becomes a Python function which can be called as ``func(a, b)`` from other part of the template and returns a rendered string

.. _guide-writing-a-transformation:

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
Also the ``dtypes`` variable is available in the template and gives access :py:mod:`~tigger.cluda.dtypes` module, and ``func`` is a module-like object containing generalizations of arithmetic functions (see :ref:`cluda-kernel-toolbox` for details).

For example, for a scaling transformation with one input, one output and one parameter the code may look like:

::

    ${o1.store}(${func.mul(i1.dtype, p1.dtype, out=o1.dtype)}(${i1.load}, ${p1}));

There is a lot of stuff going on in this single line.
First, notice that the input is loaded as ``${i1.load}``, and the parameter as ``${p1}``.
Second, since any of the ``i1`` and ``p1`` can be complex, we had to use the generic multiplication template from the ``func`` quasi-module.
The result is passed to the output by calling ``${o1.store}``.
If the transformation has several outputs, it will have several ``store`` statements.

.. _guide-writing-a-computation:

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
