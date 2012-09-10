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

    import numpy

    from tigger.helpers import *
    from tigger.core import *

    class TestComputation(Computation):

Each computation class has to define the following methods:

#.
    ::

        def _get_argnames(self):
            return ('output',), ('input1', 'input2'), ('param',)

    First, we have to specify :py:meth:`~tigger.core.Computation._get_argnames` which returns argument names for the computation.
    The arguments are split into three groups: outputs, inputs and scalar arguments.

    If you do not implement this method, you will need to implement a method that calls :py:meth:`~tigger.core.Computation._set_argnames`, which will finish initialization.
    This method has to be called prior to :py:meth:`~tigger.core.Computation.connect` and :py:meth:`~tigger.core.Computation.prepare`.
    This is only necessary if your computation class can have different number of arguments depending on some parameters.
    For an example, see the implementation of :py:class:tigger.elementwise.Elementwise`.

    Then you need to think about what values will constitute a basis for the computation.
    Basis should contain all the information to fully specify kernels, allocations and all other computation details.
    In our case, we will force all the variables have the same data type (although it is not necessary).
    In addition we will need to add the array size to the basis.

#.
    ::

        def _get_default_basis(self):
            return dict(dtype=numpy.float32, size=1)

    Method :py:meth:`~tigger.core.Computation._get_default_basis` returns a dcitionary with default values for the basis:

    In our case the computation depends only on a data type and on a size of arrays being processed.

#.
    ::

        def _get_basis_for(self, default_basis, output, input1, input2, param):
            assert output.dtype == input1.dtype
            assert output.dtype == input2.dtype
            assert output.dtype == param.dtype
            assert output.shape == input1.shape
            assert output.shape == input2.shape
            return dict(shape=output.shape, dtype=output.dtype)

    Next method to overload, :py:meth:`~tigger.core.Computation._get_basis_for`, creates a basis based on the actual parameters passed to the computation.

    :py:meth:`~tigger.core.Computation._get_basis_for` gets executed when the user calls :py:meth:`~tigger.core.Computation.prepare_for`.
    The keywords from :py:meth:`~tigger.core.Computation.prepare_for` are passed directly to :py:meth:`~tigger.core.Computation._get_basis_for`, but positional arguments may not be the same because of attached transformations.
    Therefore :py:meth:`~tigger.core.Computation._get_basis_for` gets instances of :py:meth:`~tigger.core.ArrayValue` and :py:meth:`~tigger.core.ScalarValue` as positional arguments.
    At this stage we do not care about the actual data, only its properties, namely ``shape`` and ``dtype``.

    Default basis from the previous method is passed as ``default_basis`` parameter.

#.

    ::

        def _get_argvalues(self, basis):
            return dict(
                output=ArrayValue(basis.shape, basis.dtype),
                input1=ArrayValue(basis.shape, basis.dtype),
                input2=ArrayValue(basis.shape, basis.dtype),
                param=ScalarValue(basis.dtype))

    This is the introspection method which tells what arguments (array/scalar, data types and shapes) the prepared computation expects to get.
    Again, if there are transformations attached, these values will be propagated through the tree (from roots to leaves) before returning to the user.

#.
    ::

        def _construct_operations(self, operations, basis, device_params):
            template = template_from(
                """
                <%def name='testcomp(k_output, k_input1, k_input2, k_param)'>
                ${kernel_definition}
                {
                    VIRTUAL_SKIP_THREADS;
                    int idx = virtual_global_flat_id();
                    ${k_output.ctype} result = ${k_input1.load}(idx) +
                        ${func.mul(k_input2.dtype, k_param.dtype)}(
                            ${k_input2.load}(idx), ${k_param});
                    ${k_output.store}(result, idx);
                }
                </%def>
                """)

            operations.add_kernel(template, 'testcomp', ['output', 'input1', 'input2', 'param'],
                global_size=(basis.size,), render_kwds=dict(size=basis.size))


    The last method actually creates kernels and specifies their call parameters.
    Every kernel call is based on the separate template function.
    The template can be specified as a string using :py:func:`~tigger.helpers.template_from`, or loaded as a separate file.
    Usual pattern in this case is to call it the same as the file where the computation class is defined (for example, ``testcomp.mako`` for ``testcomp.py``), and store it in some variable on module load as ``TEMPLATE = template_for(__file__)`` using :py:func:`~tigger.helpers.template_for`.

    The template function should take the same number of positional arguments as the kernel; you can view ``<%def ... >`` part as an actual kernel definition, but with the arguments being python objects containing variable metadata.
    Namely, each object has attributes ``dtype`` and ``ctype``, which contains numpy data type and C type string for the corresponding argument.
    Also, depending on whether the corresponding argument is an output array, an input array or a scalar parameter, the object can be used as ``${obj.store}(val, index)``, ``${obj.load}(index)`` or ``${obj}``.
    This will produce corresponding request to the global memory or kernel arguments.

    If you need additional device functions, they have to be specified between ``<%def ... >`` and ``${kernel_definition}`` (the latter is where the actual kernel signature will be rendered).
    Obviously, these functions can still use ``dtype`` and ``ctype`` object properties, although ``store`` and ``load`` will lead to unpredictable results (since they are rendered as macros using main kernel arguments).

    Since kernel call parameters are specified on creation, all kernel calls are rendered as CLUDA static kernels (see :py:meth:`~tigger.cluda.api.Context.compile_static`) and therefore can use all corresponding macros and functions (like :c:func:`virtual_global_flat_id` in our kernel).
    Also, they should have :c:macro:`VIRTUAL_SKIP_THREADS` at the beginning of the kernel.

    ``operations`` is a :py:class:`~tigger.core.operation.OperationRecorder` object, and ``device_params`` is a :py:class:`~tigger.cluda.api.DeviceParameters` object, which is used to optimize the computation for the specific device.

