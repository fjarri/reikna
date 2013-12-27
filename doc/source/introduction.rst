************
Introduction
************

This section contains a brief illustration of what ``reikna`` does.
For more details see :ref:`basic <tutorial-basic>` and :ref:`advanced <tutorial-advanced>` tutorials.


CLUDA
=====

CLUDA is an abstraction layer on top of PyCUDA/PyOpenCL.
Its main purpose is to separate the rest of ``reikna`` from the difference in their APIs, but it can be used by itself too for some simple tasks.

Consider the following example, which is very similar to the one from the index page on PyCUDA documentation:

.. testcode:: cluda_simple_example

    import numpy
    import reikna.cluda as cluda

    N = 256

    api = cluda.ocl_api()
    thr = api.Thread.create()

    program = thr.compile("""
    KERNEL void multiply_them(
        GLOBAL_MEM float *dest,
        GLOBAL_MEM float *a,
        GLOBAL_MEM float *b)
    {
      const SIZE_T i = get_local_id(0);
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = program.multiply_them

    a = numpy.random.randn(N).astype(numpy.float32)
    b = numpy.random.randn(N).astype(numpy.float32)
    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    dest_dev = thr.empty_like(a_dev)

    multiply_them(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    print((dest_dev.get() - a * b == 0).all())

.. testoutput:: cluda_simple_example
    :hide:

    True

If you are familiar with ``PyCUDA`` or ``PyOpenCL``, you will easily understand all the steps we have made here.
The ``cluda.ocl_api()`` call is the only place where OpenCL is mentioned, and if you replace it with ``cluda.cuda_api()`` it will be enough to make the code use CUDA.
The abstraction is achieved by using generic API module on the Python side, and special macros (:c:macro:`KERNEL`, :c:macro:`GLOBAL_MEM`, and others) on the kernel side.

The argument of :py:meth:`~reikna.cluda.api.Thread.compile` method can also be a template, which is quite useful for metaprogramming, and also used to compensate for the lack of complex number operations in CUDA and OpenCL.
Let us illustrate both scenarios by making the initial example multiply complex arrays.
The template engine of choice in ``reikna`` is `Mako <http://www.makotemplates.org>`_, and you are encouraged to read about it as it is quite useful. For the purpose of this example all we need to know is that ``${python_expression()}`` is a synthax construction which renders the expression result.

.. testcode:: cluda_template_example

    import numpy
    from numpy.linalg import norm

    from reikna import cluda
    from reikna.cluda import functions, dtypes

    N = 256
    dtype = numpy.complex64

    api = cluda.ocl_api()
    thr = api.Thread.create()

    program = thr.compile("""
    KERNEL void multiply_them(
        GLOBAL_MEM ${ctype} *dest,
        GLOBAL_MEM ${ctype} *a,
        GLOBAL_MEM ${ctype} *b)
    {
      const SIZE_T i = get_local_id(0);
      dest[i] = ${mul}(a[i], b[i]);
    }
    """, render_kwds=dict(
        ctype=dtypes.ctype(dtype),
        mul=functions.mul(dtype, dtype)))

    multiply_them = program.multiply_them

    r1 = numpy.random.randn(N).astype(numpy.float32)
    r2 = numpy.random.randn(N).astype(numpy.float32)
    a = r1 + 1j * r2
    b = r1 - 1j * r2
    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    dest_dev = thr.empty_like(a_dev)

    multiply_them(dest_dev, a_dev, b_dev, local_size=N, global_size=N)
    print(norm(dest_dev.get() - a * b) / norm(a * b) <= 1e-6)

.. testoutput:: cluda_template_example
    :hide:

    True

Note that CLUDA ``Thread`` is created by means of a static method and not using the constructor.
The constructor is reserved for more probable scenario, where we want to include some ``reikna`` functionality in a larger program, and we want it to use the existing context and stream/queue (see the :py:class:`~reikna.cluda.api.Thread` constructor).
In this case all further operations with the thread will be performed using the objects provided.

Here we have passed two values to the template: ``ctype`` (a string with C type name), and ``mul`` which is a :py:class:`~reikna.cluda.Module` object containing a single multiplication function.
The object is created by a function :py:func:`~reikna.cluda.functions.mul` which takes data types being multiplied and returns a module that was parametrized accordingly.
Inside the template the variable ``mul`` is essentially the prefix for all the global C objects (functions, structures, macros etc) from the module.
If there is only one public object in the module (which is recommended), it is a common practice to give it the name consisting just of the prefix, so that it could be called easily from the parent code.

For more information on modules, see :ref:`tutorial-modules`; the complete list of things available in CLUDA can be found in :ref:`CLUDA reference <api-cluda>`.


Computations
============

Now it's time for the main part of the functionality.
``reikna`` provides GPGPU algorithms in the form of :py:class:`~reikna.core.Computation`-based cores and :py:class:`~reikna.core.Transformation`-based plug-ins.
Computations contain the algorithm itself; examples are matrix multiplication, reduction, sorting and so on.
Transformations are parallel operations on inputs or outputs of computations, used for scaling, typecast and other auxiliary purposes.
Transformations are compiled into the main computation kernel and are therefore quite cheap in terms of performance.

As an example, we will consider the matrix multiplication.

.. testcode:: matrixmul_example

    import numpy
    from numpy.linalg import norm
    import reikna.cluda as cluda
    from reikna.linalg import MatrixMul

    api = cluda.ocl_api()
    thr = api.Thread.create()

    shape1 = (100, 200)
    shape2 = (200, 100)

    a = numpy.random.randn(*shape1).astype(numpy.float32)
    b = numpy.random.randn(*shape2).astype(numpy.float32)
    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    res_dev = thr.array((shape1[0], shape2[1]), dtype=numpy.float32)

    dot = MatrixMul(a_dev, b_dev, out_arr=res_dev)
    dotc = dot.compile(thr)
    dotc(res_dev, a_dev, b_dev)

    res_reference = numpy.dot(a, b)

    print(norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6)

.. testoutput:: matrixmul_example
    :hide:

    True

Most of the code above should be already familiar, with the exception of the creation of :py:class:`~reikna.linalg.MatrixMul` object.
The computation constructor takes two array-like objects, representing arrays that will participate in the computation.
After that the computation object has to be compiled.
The :py:meth:`~reikna.core.Computation.compile` method requires a :py:class:`~reikna.cluda.api.Thread` object, which serves as a source of data about the target API and device, and provides an execution queue.


Transformations
===============

Now imagine that you want to multiply complex matrices, but real and imaginary parts of your data are kept in separate arrays.
You could create additional kernels that would join your data into arrays of complex values, but this would require additional storage and additional calls to GPU.
Transformation API allows you to connect these transformations to the core computation --- matrix multiplication --- effectively adding the code into the main computation kernel and changing its signature.

Let us change the previous example and connect transformations to it.

.. testcode:: transformation_example

    import numpy
    from numpy.linalg import norm
    import reikna.cluda as cluda
    from reikna.core import Type
    from reikna.linalg import MatrixMul
    from reikna.transformations import combine_complex

    api = cluda.ocl_api()
    thr = api.Thread.create()

    shape1 = (100, 200)
    shape2 = (200, 100)

    a_re = numpy.random.randn(*shape1).astype(numpy.float32)
    a_im = numpy.random.randn(*shape1).astype(numpy.float32)
    b_re = numpy.random.randn(*shape2).astype(numpy.float32)
    b_im = numpy.random.randn(*shape2).astype(numpy.float32)

    arrays = [thr.to_device(x) for x in [a_re, a_im, b_re, b_im]]
    a_re_dev, a_im_dev, b_re_dev, b_im_dev = arrays

    a_type = Type(numpy.complex64, shape=shape1)
    b_type = Type(numpy.complex64, shape=shape2)
    res_dev = thr.array((shape1[0], shape2[1]), dtype=numpy.complex64)

    dot = MatrixMul(a_type, b_type, out_arr=res_dev)
    combine_a = combine_complex(a_type)
    combine_b = combine_complex(b_type)

    dot.parameter.matrix_a.connect(
        combine_a, combine_a.output, a_re=combine_a.real, a_im=combine_a.imag)
    dot.parameter.matrix_b.connect(
        combine_b, combine_b.output, b_re=combine_b.real, b_im=combine_b.imag)

    dotc = dot.compile(thr)

    dotc(res_dev, a_re_dev, a_im_dev, b_re_dev, b_im_dev)

    res_reference = numpy.dot(a_re + 1j * a_im, b_re + 1j * b_im)

    print(norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6)

.. testoutput:: transformation_example
    :hide:

    True

We have used a pre-created transformation :py:func:`~reikna.transformations.combine_complex` from :py:mod:`reikna.transformations` for simplicity; developing a custom transformation is also possible and described in :ref:`tutorial-advanced-transformation`.
From the documentation we know that it transforms two inputs into one output; therefore we need to attach it to one of the inputs of ``dot`` (identified by its name), and provide names for two new inputs.

Names to attach to are obtained from the documentation for the particular computation; for :py:class:`~reikna.linalg.MatrixMul` these are ``out``, ``a`` and ``b``.

In the current example we have attached the transformations to both inputs.
Note that the computation has a new signature now, and the compiled ``dot`` object now works with split complex numbers.
