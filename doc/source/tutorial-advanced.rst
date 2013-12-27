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
Transformations are based on the class :py:class:`~reikna.core.Transformation`, and are very similar to :py:class:`~reikna.algorithms.PureParallel` instances, with some additional limitations.

Let us consider a (not very useful, but quite involved) example:

::

    tr = Transformation(
        [
            Parameter('out1', Annotation(Type(numpy.float32, shape=100), 'o')),
            Parameter('out2', Annotation(Type(numpy.float32, shape=80), 'o')),
            Parameter('in1', Annotation(Type(numpy.float32, shape=100), 'i')),
            Parameter('in2', Annotation(Type(numpy.float32, shape=100), 'i')),
            Parameter('param', Annotation(Type(numpy.float32))),
        ],
        """
        VSIZE_T idx = ${idxs[0]};
        float i1 = ${in1.load_same};
        float i2 = ${in2.load_idx}(100 - idx) * ${param};
        ${out1.store_same}(i1);
        if (idx < 80)
            ${out2.store_same}(i2);
        """,
        connectors=['in1', 'out1'])

**Connectors.**
A transformation gets activated when the main computation attempts to load some value from some index in global memory, or store one to some index.
This index is passed to the transformation attached to the corresponding parameter, and used to invoke loads/stores either without changes (to perform strictly elementwise operations), or, possibly, with some changes (as the example illustrates).

If some parameter is only queried once, and only using ``load_same`` or ``store_same``, it is called a *connector*, which means that it can be used to attach the transformation to a computation.
Currently connectors cannot be detected automatically, so it is the responsibility of the user to provide a list of them to the constructor.
By default all parameters are considered to be connectors.

**Shape changing.**
Parameters in transformations are typed, and it is possible to change data type or shape of a parameter the transformation is attached to.
In our example ``out2`` has length 80, so the current index is checked before the output to make sure there is no out of bounds access.

**Parameter objects.**
The transformation example above has some hardcoded stuff, for example the type of parameters (``float``), or their shapes (``100`` and ``80``).
These can be accessed from argument objects ``out1``, ``in1`` etc; they all have the type :py:class:`~reikna.core.transformation.KernelParameter`.
In addition, the transformation code gets an :py:class:`~reikna.core.Indices` object with the name ``idxs``, which allows one to manipulate index names directly.


.. _tutorial-advanced-computation:

Writing a computation
=====================

A computation must derive :py:class:`~reikna.core.Computation`.
As an example, let us create a computation which calculates ``output = input1 + input2 * param``.

Defining a class:

::

    import numpy

    from reikna.helpers import *
    from reikna.core import *

    class TestComputation(Computation):

Each computation class has to define the constructor, and the plan building callback.

**Constructor.**
:py:class:`~reikna.core.Computation` constructor takes a list of computation parameters, which the deriving class constructor has to create according to arguments passed to it.
You will often need :py:class:`~reikna.core.Type` objects, which can be extracted from arrays, scalars or other :py:class:`~reikna.core.Type` objects with the help of :py:meth:`~reikna.core.Type.from_value` (or they can be passed straight to :py:class:`~reikna.core.Annotation`) which does the same thing.

::

    def __init__(self, arr, coeff):
        assert len(arr.shape) == 1
        Computation.__init__(self, [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('input1', Annotation(arr, 'i')),
            Parameter('input2', Annotation(arr, 'i')),
            Parameter('param', Annotation(coeff))])

In addition to that, the constructor can create some internal state which will be used by the plan builder.

**Plan builder.**
The second method is called when the computation is being compiled, and has to fill and return the computation plan --- a sequence of kernel calls, plus maybe some temporary or persistent internal allocations its kernels use.
In addition, the plan can include calls to nested computations.

The method takes two predefined positional parameters, plus :py:class:`~reikna.core.computation.KernelArgument` objects corresponding to computation parameters.
The ``plan_factory`` is a callable that creates a new :py:class:`~reikna.core.computation.ComputationPlan` object (in some cases you may want to recreate the plan, for example, if the workgroup size you were using turned out to be too big), and ``device_params`` is a :py:class:`~reikna.cluda.api.DeviceParameters` object, which is used to optimize the computation for the specific device.
The method must return a filled :py:class:`~reikna.core.computation.ComputationPlan` object.

For our example we only need one action, which is the execution of an elementwise kernel:

::

    def _build_plan(self, plan_factory, device_params, output, input1, input2, param):
        plan = plan_factory()

        template = template_from(
            """
            <%def name='testcomp(kernel_declaration, k_output, k_input1, k_input2, k_param)'>
            ${kernel_declaration}
            {
                VIRTUAL_SKIP_THREADS;
                const VSIZE_T idx = virtual_global_id(0);
                ${k_output.ctype} result =
                    ${k_input1.load_idx}(idx) +
                    ${mul}(${k_input2.load_idx}(idx), ${k_param});
                ${k_output.store_idx}(idx, result);
            }
            </%def>
            """)

        plan.kernel_call(
            template.get_def('testcomp'),
            [output, input1, input2, param],
            global_size=output.shape,
            render_kwds=dict(mul=functions.mul(input2.dtype, param.dtype)))

        return plan

Every kernel call is based on the separate ``Mako`` template def.
The template can be specified as a string using :py:func:`~reikna.helpers.template_def`, or loaded as a separate file.
Usual pattern in this case is to call the template file same as the file where the computation class is defined (for example, ``testcomp.mako`` for ``testcomp.py``), and store it in some variable on module load using :py:func:`~reikna.helpers.template_for` as ``TEMPLATE = template_for(__file__)``.

The template function should take the same number of positional arguments as the kernel plus one; you can view ``<%def ... >`` part as an actual kernel definition, but with the arguments being :py:class:`~reikna.core.transformation.KernelParameter` objects containing parameter metadata.
The first argument will contain the string with the kernel declaration.

Also, depending on whether the corresponding argument is an output array, an input array or a scalar parameter, the object can be used as ``${obj.store_idx}(index, val)``, ``${obj.load_idx}(index)`` or ``${obj}``.
This will produce the corresponding request to the global memory or kernel arguments.

If you need additional device functions, they have to be specified between ``<%def ... >`` and ``${kernel_declaration}``.
Obviously, these functions can still use ``dtype`` and ``ctype`` object properties, although ``store_idx`` and ``load_idx`` will most likely result in compilation error (since they are rendered as macros using main kernel arguments).

Since kernel call parameters (``global_size`` and ``local_size``) are specified on creation, all kernel calls are rendered as CLUDA static kernels (see :py:meth:`~reikna.cluda.api.Thread.compile_static`) and therefore can use all the corresponding macros and functions (like :c:func:`virtual_global_flat_id` in our kernel).
Also, they must have :c:macro:`VIRTUAL_SKIP_THREADS` at the beginning of the kernel which remainder threads (which can be present, for example, if the workgroup size is not a multiple of the global size).
