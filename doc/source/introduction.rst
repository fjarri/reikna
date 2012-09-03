Introduction
============

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

    multiply_them(dest_dev, a_dev, b_dev, local_size=(N,), global_size=(N,))
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

You may have notice that CLUDA context is created by means of a static method and not using the constructor.
The constructor is reserved for more usual scenario, where you want to include some Tigger functionality in your bigger script, and want it to use the existing context and stream/queue.
The ``Context`` constructor takes the PyCuda/PyOpenCL context and, optionally, the ``Stream``/``CommandQueue`` object as a ``stream`` parameter.
All further operations with the Tigger context will be performed using given context and stream.
If ``stream`` is not given, an internal stream will be created.

For the complete list of things available in CLUDA, please consult :ref:`CLUDA reference <cluda-reference>`.


Computations, user point of view
--------------------------------

Now it's time for the main part of the functionality.
Tigger provides GPGPU algorithms in the form of ``Computation`` classes and ``Transformation`` objects.
Computations contain the algorithm itself; examples are matrix multiplication, reduction, sorting and so on.
Transformations are elementwise operations on inputs/outputs of computations, used for scaling, typecast and other auxiliary purposes.
Transformations are compiled into the main computation kernel and are therefore quite cheap in terms of performance.

As an example, we will consider the matrix multiplication.

::

    import numpy
    from numpy.linalg import norm
    import tigger.cluda as cluda
    from tigger.matrixmul import MatrixMul

    api = cluda.cuda_api()
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

    print norm(ctx.from_device(res_dev) - res_reference) / norm(res_reference)

Most of the code above should be already familiar, with the exception of the creation of ``MatrixMul`` object.
As any other class derived from ``Computation``, it requires Tigger context as a constructor argument.
The context serves as a source of data about the target API and device, and provides an execution stream.

After the creation the object has to be prepared.
It does not happen automatically, since there are two preparation methods, and since it is pointless to compile a kernel that will not be used anyway.
First method can be seen in the example above.
We know (from the documentation) that ``MatrixMul.__call__()`` takes three array parameters, and we ask it to prepare itself to properly handle arrays ``res_dev``, ``a_dev`` and ``b_dev`` when they are passed to it.
Alternatively, this information can be obtained from console by examining ``signature`` property of the object:

::

    >>> dot = MatrixMul(ctx)
    >>> dot.signature
    [('C', ArrayValue(None,None)), ('A', ArrayValue(None,None)), ('B', ArrayValue(None,None))]

The second method is directly specify the parameter basis --- a dictionary of parameters which define all the internal preparations to be done (when ``prepare_for()`` is called, these are derived from its arguments).
Again, looking at the reference, we can see that ``MatrixMul`` has a dozen of parameters, the most important being input and output arrays types and sizes.
If, for some reason, actual arrays are not available at the time of preparation, ``prepare()`` with necessary keyword arguments can be called instead.


Transformations
---------------

Now imagine that you want to multiply complex matrices, but real and imaginary parts of your data are kept in separate arrays.
You could create elementwise kernels that would join your data into arrays of complex values, but this would require additional storage and additional calls to GPU.
Transformation API allows you to connect these transformations to the core computation --- matrix multiplication --- effectively adding the code into the main computation kernel and changing its signature.

Let us change the previous example and connect transformations to it.

::

    import numpy
    from numpy.linalg import norm
    import tigger.cluda as cluda
    import tigger.cluda.dtypes as dtypes
    from tigger.matrixmul import MatrixMul
    from tigger import Transformation

    api = cluda.cuda_api()
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

    split_to_interleaved = Transformation(
        load=2, store=1,
        derive_o_from_is=lambda l1, l2: [dtypes.complex_for(l1)],
        derive_is_from_o=lambda s1: ([dtypes.real_for(s1), dtypes.real_for(s1)], []),
        code="""
            ${store.s1}(${dtypes.complex_ctr(numpy.complex64)}(${load.l1}, ${load.l2}));
        """)
    dot.connect(split_to_interleaved, 'A', ['A_re', 'A_im'])
    dot.connect(split_to_interleaved, 'B', ['B_re', 'B_im'])
    dot.prepare_for(res_dev, a_re_dev, a_im_dev, b_re_dev, b_im_dev)

    dot(res_dev, a_re_dev, a_im_dev, b_re_dev, b_im_dev)

    res_reference = numpy.dot(a_re + 1j * a_im, b_re + 1j * b_im)

    print norm(ctx.from_device(res_dev) - res_reference) / norm(res_reference)

This requires a bit of explanation.
First, we create a transformation ``split_to_interleaved`` with two inputs and one output.
Next two parameters are type derivation functions --- they will be used internally to derive basis from ``prepare_for()`` arguments, and signature types from the basis, respectively.
Code is a small Mako template, which uses two inputs ``${load.l1}`` and ``${load.l2}``, passes them to the complex number constructor and stores the result in ``${store.s1}``.
This transformation is then attached to endpoints ``A`` and ``B`` --- the input values of basic ``MatrixMul`` computation.
Finally, we call ``prepare_for()`` which now has a new signature, and the resulting ``dot`` object now works with split complex numbers.
