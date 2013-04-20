.. _tutorial-advanced:

*************************
Tutorial: advanced topics
*************************

This tutorial goes into more detail about the internals of computations and transformations, describing how to write them.

Mako basics
===========

``Reikna`` uses `Mako <http://makotemplates.org>`_ extensively as a templating engine for transformations and computations.
For the purpose of this tutorial you only need to know several things about the synthax:

* Most of Mako synthax is plain Python, with the set of global variables specified externally by the code doing the template rendering
* ``${expr}`` evaluates Python expression ``expr``, calls ``str()`` on the result and puts it into the text
* a pair of ``<%`` and ``%>`` executes Python code inside, which may introduce some local variables
* a pair of ``<%def name="func(a, b)">`` and ``</%def>`` defines a template function, which actually becomes a Python function which can be called as ``func(a, b)`` from the other part of the template and returns a rendered string


.. _tutorial-advanced-transformation:

Writing a transformation
========================

Some common transformations are already available from :py:mod:`~reikna.transformations` module.
But you can create a custom one if you need to.
Transformations are based on the class :py:class:`~reikna.core.Transformation`.
Its constructor has three major groups of parameters.

First, ``outputs``, ``inputs`` and ``scalars`` contain lists of names for corresponding transformation arguments.
Alternatively, you may just pass integers; in that case the names will be generated to be ``i1``, ``i2``, ..., ``o1``, ``o2``, ..., ``s1``, ``s2``, ...

Second, ``derive_o_from_is`` and ``derive_i_from_os`` options take functions that perform type derivation.
This happens when ``prepare_for`` is called; first function will be used to propagate types from leaf inputs to base inputs, and the second one to propagate type from leaf outputs to base outputs.
If the transformation has more than one output or more than one input, and, therefore, cannot be connected to the output or input argument, respectively, the corresponding function should not be supplied.
On the other hand, if the function is not supplied, but is required, the fallback is the :py:func:`~reikna.cluda.dtypes.result_type`.

The format of required functions is the following (here ``iN``, ``oN`` and ``sN`` are :py:class:`numpy.dtype` objects):

* ``derive_o_from_is(i1, ..., s1, ...)``, returns the :py:class:`numpy.dtype` for ``o1``.
* ``derive_i_from_os(o1, ..., s1, ...)``, returns the :py:class:`numpy.dtype` for ``i1``.

The last part of the constructor is a ``code`` parameter.
It is a string with the Mako template which describes the transformation.
Variables ``i1``, ..., ``o1``, ..., ``s1``, ... are available in the template and help specify load and store actions for inputs, outputs and parameters, and also to obtain their data types.
Each of these variables has attributes ``dtype`` (contains the :py:class:`numpy.dtype`), ``ctype`` (contains a string with corresponding C type) and either one of ``load`` (for inputs), ``store`` (for outputs) and ``__str__`` (for scalar parameters).
``${i1.load}`` can be used as a variable, and ``${o1.store}(val)`` as a function that takes one variable.
Also the ``dtypes`` variable is available in the template, providing access :py:mod:`~reikna.cluda.dtypes` module, and ``func`` is a module-like object containing generalizations of arithmetic functions (see :ref:`cluda-kernel-toolbox` for details).

For example, for a scaling transformation with one input, one output and one parameter the code may look like:

::

    ${o1.store}(${func.mul(i1.dtype, s1.dtype, out=o1.dtype)}(${i1.load}, ${s1}));

There is a lot of stuff going on in this single line.
First, notice that the input is loaded as ``${i1.load}``, and the parameter as ``${s1}``.
Second, since any of the ``i1`` and ``s1`` can be complex, we had to use the generic multiplication template from the ``func`` quasi-module.
The result is passed to the output by calling ``${o1.store}``.
If the transformation has several outputs, it will have several ``store`` statements.
Since the ``code`` parameter will be inserted into a function, you can safely create temporary variables if you need to.


.. _tutorial-advanced-computation:

Writing a computation
=====================

A computation must derive :py:class:`~reikna.core.Computation` class and implement several methods.
As an example, let us create a computation which calculates ``output = input1 + input2 * param``.

Defining a class:

::

    import numpy

    from reikna.helpers import *
    from reikna.core import *

    class TestComputation(Computation):

Each computation class has to define the following methods:

#.  First, we have to specify :py:meth:`~reikna.core.Computation._get_argnames` which returns argument names for the computation.
    The arguments are split into three groups: outputs, inputs and scalar arguments.

    ::

        def _get_argnames(self):
            return ('output',), ('input1', 'input2'), ('param',)

    If you do not implement this method, you will need to implement a method that calls :py:meth:`~reikna.core.Computation._set_argnames`, which will finish initialization.
    When the computation object is created, this method has to be called prior to any calls to ``connect`` or ``prepare_for``.
    This is only necessary if your computation class can have different number of arguments depending on some parameters.
    For an example, see the implementation of :py:class:reikna.elementwise.Elementwise`.

#.  Then you need to think about what values will constitute a basis for the computation.
    Basis should contain all the information necessary to specify kernels, allocations and all other computation details.
    In our case, we will force all the variables to have the same data type (although it is not necessary).
    In addition we will need to add the array shape to the basis.
    The method :py:meth:`~reikna.core.Computation._get_basis_for`, gets executed when the user calls ``prepare_for`` and creates a basis based on the arguments and keywords passed to it.

    ::

        def _get_basis_for(self, output, input1, input2, param):
            assert output.dtype == input1.dtype == input2.dtype == param.dtype
            assert output.shape == input1.shape == input2.shape
            return dict(shape=output.shape, dtype=output.dtype)

    The keywords from ``prepare_for`` are passed directly to :py:meth:`~reikna.core.Computation._get_basis_for`, but positional arguments may not be the same because of attached transformations.
    Therefore :py:meth:`~reikna.core.Computation._get_basis_for` gets instances of :py:class:`~reikna.core.ArrayValue` and :py:class:`~reikna.core.ScalarValue` as positional arguments.
    At this stage we do not care about the actual data, only its properties, like shape and data type.

#.  Next method tells what arguments (array/scalar, data types and shapes) the prepared computation expects to get.
    This method is used in some internal algorithms.

    ::

        def _get_argvalues(self, basis):
            return dict(
                output=ArrayValue(basis.shape, basis.dtype),
                input1=ArrayValue(basis.shape, basis.dtype),
                input2=ArrayValue(basis.shape, basis.dtype),
                param=ScalarValue(basis.dtype))

#.  The last method actually specifies the actions to be done by the computation.
    These include kernel calls, allocations and calls to nested computations.
    The method takes two parameters: ``basis`` is a basis created by :py:meth:`~reikna.core.Computation._get_basis_for`, and ``device_params`` is a :py:class:`~reikna.cluda.api.DeviceParameters` object, which is used to optimize the computation for the specific device.
    It must return a filled :py:class:`~reikna.core.operation.OperationRecorder` object.

    For our example we only need one action, which is the execution of an elementwise kernel:

    ::

        def _construct_operations(self, basis, device_params):
            operations = self._get_operation_recorder()
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
                    ${k_output.store}(idx, result);
                }
                </%def>
                """)

            operations.add_kernel(template.get_def('testcomp'),
                ['output', 'input1', 'input2', 'param'],
                global_size=basis.shape)
            return operations

    Every kernel call is based on the separate ``Mako`` template function.
    The template can be specified as a string using :py:func:`~reikna.helpers.template_func`, or loaded as a separate file.
    Usual pattern in this case is to call the template file same as the file where the computation class is defined (for example, ``testcomp.mako`` for ``testcomp.py``), and store it in some variable on module load using :py:func:`~reikna.helpers.template_for` as ``TEMPLATE = template_for(__file__)``.

    The template function should take the same number of positional arguments as the kernel; you can view ``<%def ... >`` part as an actual kernel definition, but with the arguments being python objects containing variable metadata.
    Namely, every such object has attributes ``dtype`` and ``ctype``, which contain :py:class:`numpy.dtype` object and C type string for the corresponding argument.
    Also, depending on whether the corresponding argument is an output array, an input array or a scalar parameter, the object can be used as ``${obj.store}(val, index)``, ``${obj.load}(index)`` or ``${obj}``.
    This will produce corresponding request to the global memory or kernel arguments.

    If you need additional device functions, they have to be specified between ``<%def ... >`` and ``${kernel_definition}`` (the latter is where the actual kernel signature will be rendered).
    Obviously, these functions can still use ``dtype`` and ``ctype`` object properties, although ``store`` and ``load`` will lead to unpredictable results (since they are rendered as macros using main kernel arguments).

    Since kernel call parameters (``global_size`` and ``local_size``) are specified on creation, all kernel calls are rendered as CLUDA static kernels (see :py:meth:`~reikna.cluda.api.Context.compile_static`) and therefore can use all the corresponding macros and functions (like :c:func:`virtual_global_flat_id` in our kernel).
    Also, they must have :c:macro:`VIRTUAL_SKIP_THREADS` at the beginning of the kernel.
