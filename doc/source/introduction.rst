************
Introduction
************

This section contains a brief illustration of what ``Tigger`` does.
For more details see :ref:`basic <tutorial-basic>` and :ref:`advanced <tutorial-advanced>` tutorials.


CLUDA
=====

CLUDA is an abstraction layer on top of PyCuda/PyOpenCL.
Its main purpose is to separate the rest of ``Tigger`` from the difference in their APIs, but it can be used by itself too for some simple tasks.

Consider the following example, which is very similar to the one from the index page on PyCuda documentation:

.. testcode:: cluda_simple_example

    import numpy
    import tigger.cluda as cluda

    N = 256

    api = cluda.ocl_api()
    ctx = api.Context.create()

    module = ctx.compile("""
    KERNEL void multiply_them(
        GLOBAL_MEM float *dest,
        GLOBAL_MEM float *a,
        GLOBAL_MEM float *b)
    {
      const int i = get_local_id(0);
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = module.multiply_them

    a = numpy.random.randn(N).astype(numpy.float32)
    b = numpy.random.randn(N).astype(numpy.float32)
    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    dest_dev = ctx.empty_like(a_dev)

    multiply_them(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    print (dest_dev.get() - a * b == 0).all()

.. testoutput:: cluda_simple_example
    :hide:

    True

If you are familiar with PyCuda or PyOpenCL, you will easily understand all the steps we have done here.
The ``cluda.ocl_api()`` call is the only place where OpenCL is mentioned, and if you replace it with ``cluda.cuda_api()`` it will be enough to make the code use CUDA.
The abstraction is achieved by using generic API module on the Python side, and special macros (:c:macro:`KERNEL`, :c:macro:`GLOBAL_MEM`, and others) on the kernel side.

The argument of :py:meth:`~tigger.cluda.api.Context.compile` method can also be a template, which is quite useful for metaprogramming, and also used to compensate for the lack of complex number operations in CUDA and OpenCL.
Let us illustrate both scenarios by making the initial example multiply complex arrays.
The template engine of choice in ``Tigger`` is `Mako <http://www.makotemplates.org>`_, and you are encouraged to read about it as it is quite useful. For the purpose of this example all we need to know is that ``${python_expression()}`` is a synthax construction which renders the expression result.

.. testcode:: cluda_template_example

    import numpy
    from numpy.linalg import norm
    import tigger.cluda as cluda
    import tigger.cluda.dtypes as dtypes

    N = 256
    dtype = numpy.complex64

    api = cluda.ocl_api()
    ctx = api.Context.create()

    module = ctx.compile("""
    KERNEL void multiply_them(
        GLOBAL_MEM ${ctype} *dest,
        GLOBAL_MEM ${ctype} *a,
        GLOBAL_MEM ${ctype} *b)
    {
      const int i = get_local_id(0);
      dest[i] = ${func.mul(dtype, dtype)}(a[i], b[i]);
    }
    """, render_kwds=dict(dtype=dtype, ctype=dtypes.ctype(dtype)))

    multiply_them = module.multiply_them

    r1 = numpy.random.randn(N).astype(numpy.float32)
    r2 = numpy.random.randn(N).astype(numpy.float32)
    a = r1 + 1j * r2
    b = r1 - 1j * r2
    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    dest_dev = ctx.empty_like(a_dev)

    multiply_them(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    print norm(dest_dev.get() - a * b) / norm(a * b) <= 1e-6

.. testoutput:: cluda_template_example
    :hide:

    True

Here we passed ``dtype`` and ``ctype`` values to the template, and used ``dtype`` to get the complex number multiplication function (``func`` is one of the "built-in" values that are available in CLUDA templates).
Alternatively, we could call :py:func:`dtypes.ctype() <tigger.cluda.dtypes.ctype>` inside the template, as :py:mod:`~tigger.cluda.dtypes` module is available there too.

Note that CLUDA context is created by means of a static method and not using the constructor.
The constructor is reserved for more probable scenario, where we want to include some ``Tigger`` functionality in a larger program, and we want it to use the existing context and stream/queue.
The :py:class:`~tigger.cluda.api.Context` constructor takes the PyCuda/PyOpenCL context and, optionally, the ``Stream``/``CommandQueue`` object as a ``queue`` parameter.
All further operations with the ``Tigger`` context will be performed using the objects provided.
If ``queue`` is not given, an internal one will be created.

For the complete list of things available in CLUDA, please consult the :ref:`CLUDA reference <api-cluda>`.


Computations
============

Now it's time for the main part of the functionality.
``Tigger`` provides GPGPU algorithms in the form of :py:class:`~tigger.core.Computation`-based cores and :py:class:`~tigger.core.Transformation`-based plug-ins.
Computations contain the algorithm itself; examples are matrix multiplication, reduction, sorting and so on.
Transformations are elementwise operations on inputs or outputs of computations, used for scaling, typecast and other auxiliary purposes.
Transformations are compiled into the main computation kernel and are therefore quite cheap in terms of performance.

As an example, we will consider the matrix multiplication.

.. testcode:: matrixmul_example

    import numpy
    from numpy.linalg import norm
    import tigger.cluda as cluda
    from tigger.matrixmul import MatrixMul

    api = cluda.ocl_api()
    ctx = api.Context.create()

    shape1 = (100, 200)
    shape2 = (200, 100)

    a = numpy.random.randn(*shape1).astype(numpy.float32)
    b = numpy.random.randn(*shape2).astype(numpy.float32)
    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    res_dev = ctx.allocate((shape1[0], shape2[1]), dtype=numpy.float32)

    dot = MatrixMul(ctx).prepare_for(res_dev, a_dev, b_dev)
    dot(res_dev, a_dev, b_dev)

    res_reference = numpy.dot(a, b)

    print norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6

.. testoutput:: matrixmul_example
    :hide:

    True

Most of the code above should be already familiar, with the exception of the creation of :py:class:`~tigger.matrixmul.MatrixMul` object.
As any other class derived from :py:class:`~tigger.core.Computation`, it requires ``Tigger`` context as a constructor argument.
The context serves as a source of data about the target API and device, and provides an execution queue.

Before usage the object has to be prepared.
It does not happen in the constructor, since the transformations may be connected after that, and they would invalidate previous preparation.
The preparation consists of passing to the :py:meth:`~tigger.core.Computation.prepare_for` array and scalar arguments we will use to call the computation (or stub :py:class:`~tigger.core.ArrayValue` and :py:class:`~tigger.core.ScalarValue` objects, if real arrays are not available at preparation time), along with some optional keyword arguments.
The list of required positional and keyword arguments for any computation is specified in its documentation; for :py:class:`~tigger.matrixmul.MatrixMul` it is :py:class:`MatrixMul.prepare_for() <tigger.matrixmul.MatrixMul.prepare_for>`.

From the documentation we know that we need three array parameters, and we ask :py:class:`~tigger.matrixmul.MatrixMul` to prepare itself to handle arrays ``res_dev``, ``a_dev`` and ``b_dev`` when they are passed to it.

After the preparation we can use the object as a callable, passing it arrays and scalars with the same data types and shapes we used to prepare the computation.


Transformations
===============

Now imagine that you want to multiply complex matrices, but real and imaginary parts of your data are kept in separate arrays.
You could create elementwise kernels that would join your data into arrays of complex values, but this would require additional storage and additional calls to GPU.
Transformation API allows you to connect these transformations to the core computation --- matrix multiplication --- effectively adding the code into the main computation kernel and changing its signature.

Let us change the previous example and connect transformations to it.

.. testcode:: transformation_example

    import numpy
    from numpy.linalg import norm
    import tigger.cluda as cluda
    from tigger.matrixmul import MatrixMul
    from tigger.transformations import combine_complex

    api = cluda.ocl_api()
    ctx = api.Context.create()

    shape1 = (100, 200)
    shape2 = (200, 100)

    a_re = numpy.random.randn(*shape1).astype(numpy.float32)
    a_im = numpy.random.randn(*shape1).astype(numpy.float32)
    b_re = numpy.random.randn(*shape2).astype(numpy.float32)
    b_im = numpy.random.randn(*shape2).astype(numpy.float32)
    a_re_dev, a_im_dev, b_re_dev, b_im_dev = [ctx.to_device(x) for x in [a_re, a_im, b_re, b_im]]

    res_dev = ctx.allocate((shape1[0], shape2[1]), dtype=numpy.complex64)

    dot = MatrixMul(ctx)
    dot.connect(combine_complex(), 'a', ['a_re', 'a_im'])
    dot.connect(combine_complex(), 'b', ['b_re', 'b_im'])
    dot.prepare_for(res_dev, a_re_dev, a_im_dev, b_re_dev, b_im_dev)

    dot(res_dev, a_re_dev, a_im_dev, b_re_dev, b_im_dev)

    res_reference = numpy.dot(a_re + 1j * a_im, b_re + 1j * b_im)

    print norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6

.. testoutput:: transformation_example
    :hide:

    True

We have used a pre-created transformation :py:func:`~tigger.transformations.combine_complex` from :py:mod:`tigger.transformations` for simplicity; developing a custom transformation is also possible and described in :ref:`guide-writing-a-transformation`.
From the documentation we know that it transforms two inputs into one output; therefore we need to attach it to one of the inputs of ``dot`` (identified by its name), and provide names for two new inputs.

Names to attach to are obtained from the documentation for the particular computation. By convention they are the same as the names of positional arguments to :py:meth:`~tigger.core.Computation.prepare_for`; for :py:class:`~tigger.matrixmul.MatrixMul` these are ``out``, ``a`` and ``b``.

In the current example we have attached the transformations to both inputs.
Note that ``prepare_for`` has a new signature now, and the resulting ``dot`` object now works with split complex numbers.
