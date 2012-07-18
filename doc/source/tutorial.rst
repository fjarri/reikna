Tutorial
========

This section contains brief illustration of what Tigger does.
For detailed information see corresponding reference pages.

CLUDA basics
------------

CLUDA is an abstraction layer on top of PyCuda/PyOpenCL.
Its main purpose is to separate the rest of Tigger from the difference in their APIs, but it can be used by itself too for some simple tasks.

Consider the following example, which is very similar to the one from the index page on PyCuda documentation:

::

    import numpy
    import tigger.cluda as cluda

    N = 256

    api = cluda.cuda_api()
    ctx = api.Context.create()

    module = ctx.compile("""
    KERNEL void multiply_them(
        GLOBAL_MEM float *dest,
        GLOBAL_MEM float *a,
        GLOBAL_MEM float *b)
    {
      const int i = LID_0;
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = module.multiply_them

    a = numpy.random.randn(N).astype(numpy.float32)
    b = numpy.random.randn(N).astype(numpy.float32)
    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    dest_dev = ctx.empty_like(a_dev)

    multiply_them(dest_dev, a_dev, b_dev, block=(N,1,1), grid=(1,1))
    print ctx.from_device(dest_dev) - a * b

If you are familiar with PyCuda or PyOpenCL, you will easily understand all the steps we have done here.
The ``cluda.cuda_api()`` call is the only place where CUDA is mentioned, and if you replace it with ``cluda.ocl_api()`` it will be enough to make the code use OpenCL.
The abstraction is achieved by using generic API module on Python side, and special macros (``KERNEL``, ``GLOBAL_MEM`` and others) on kernel side.

The argument of ``compile`` method can also be a template, which is quite useful for metaprogramming, and also used to compensate for the lack of complex number operations in OpenCL.
Let us illustrate both scenarios by making the initial example multiply complex arrays.
The template engine of choice in Tigger is `Mako <http://www.makotemplates.org>`_, and you are encouraged to read about it as it is quite useful. For the purpose of this tutorial all we need to know is that its synthax is ``${python_expression()}``, which renders the expression result.

::

    import numpy
    import tigger.cluda as cluda
    import tigger.cluda.dtypes as dtypes

    N = 256
    dtype = numpy.complex64

    api = cluda.cuda_api()
    ctx = api.Context.create()

    module = ctx.compile("""
    KERNEL void multiply_them(
        GLOBAL_MEM ${ctype} *dest,
        GLOBAL_MEM ${ctype} *a,
        GLOBAL_MEM ${ctype} *b)
    {
      const int i = LID_0;
      dest[i] = ${func.mul(dtype, dtype)}(a[i], b[i]);
    }
    """, dtype=dtype, ctype=dtypes.ctype(dtype))

    multiply_them = module.multiply_them

    r1 = numpy.random.randn(N).astype(numpy.float32)
    r2 = numpy.random.randn(N).astype(numpy.float32)
    a = r1 + 1j * r2
    b = r1 - 1j * r2
    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    dest_dev = ctx.empty_like(a_dev)

    multiply_them(dest_dev, a_dev, b_dev, block=(N,1,1), grid=(1,1))
    print ctx.from_device(dest_dev) - a * b

Here we passed ``dtype`` and ``ctype`` values to the template, and used ``dtype`` to get the complex number multiplication function (``func`` is one of the "built-in" values that are available in CLUDA templates).
Alternatively, we could call ``dtypes.ctype()`` inside the template, as ``dtypes`` module is available there too.

For the complete list of things available in CLUDA, please consult :ref:`CLUDA reference <cluda-reference>`.


Computations, user point of view
--------------------------------

As an example, we will consider the matrix multiplication.

Transformations
---------------

Now let us assume you multiply complex matrices, but real and imaginary parts of your data is kept in separate arrays.
You could create elementwise kernels that would join your data into arrays of complex values, but this would require additional storage and additional calls to GPU.
Transformation API allows you to connect these transformations to the core computation --- matrix multiplication --- effectively adding the code into the main computation kernel and changing its signature.

- Elementwise pre- and post-processing can be attached to any kernel (derived from Computation class).
- Pre-processing is invoked when kernel reads from memory, and post-processing is invoked when kernel writes to memory.
  Pre-processing has to have only one output value, and post-processing has to have only one input value.
- The transformations are strictly elementwise (the user is limited by {store} and {load} macros, which do not let him specify the index).
- They can change variable types as long as there is a function that derives output type from input types (for load) or input types from output types (for store); by default these types are equal.
  Each transformation has to have both type derivations from input to output and from output to input.
  For example, if user calls prepare(), we will need to derive types "inside out", and with prepare_for() the derivation will go "outside-in".
  The library should check that all used types are actually supported by the videocard.
- Since the processing mechanism does not let us change the call signature (like adding out=None to opt for the result array being allocated during the call), we will have to have bot low-level call (with autogenerated signature) and high-level call (with maybe more convenient, but less flexible signature).
  Autogenerated signature contains only *args, with new parameters added to the end, and new arrays added in place of the ones they are generated from.
  If there are repetitions in the arg list, only the first encounter is left.

When computation has some processing attached to it, its signature changes.
