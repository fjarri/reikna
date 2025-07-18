************
Introduction
************

This section contains a brief illustration of what ``reikna`` does.
For more details see :ref:`basic <tutorial-basic>` and :ref:`advanced <tutorial-advanced>` tutorials.


Grunnur
=======

``grunnur`` is a foundation layer handling GPU abstraction and various helpers (virtual allocations, templates, arrays and so on).
Refer to `its documentation <https://grunnur.readthedocs.io/en/latest/>`_ for more details.
In particular, you will be creating contexts, queues/streams, and arrays with ``grunnur``.


Computations
============

This is the core concept of ``reikna`` functionality.
``reikna`` provides GPGPU algorithms in the form of :py:class:`~reikna.core.Computation`-based cores and :py:class:`~reikna.core.Transformation`-based plug-ins.
Computations contain the algorithm itself; examples are matrix multiplication, reduction, sorting and so on.
Transformations are parallel operations on inputs or outputs of computations, used for scaling, typecast and other auxiliary purposes.
Transformations are compiled into the main computation kernel and are therefore quite cheap in terms of performance.

As an example, we will consider the matrix multiplication.

.. testcode:: matrixmul_example

    import numpy
    from numpy.linalg import norm
    from grunnur import API, Context, Queue, Array
    from reikna.linalg import MatrixMul

    context = Context.from_devices([API.any().platforms[0].devices[0]])
    queue = Queue(context.device)

    shape1 = (100, 200)
    shape2 = (200, 100)

    a = numpy.random.randn(*shape1).astype(numpy.float32)
    b = numpy.random.randn(*shape2).astype(numpy.float32)
    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    res_dev = Array.empty(queue.device, (shape1[0], shape2[1]), dtype=numpy.float32)

    dot = MatrixMul(a_dev, b_dev, out_arr_t=res_dev)
    dotc = dot.compile(queue.device)
    dotc(queue, res_dev, a_dev, b_dev)

    res_reference = numpy.dot(a, b)

    print(norm(res_dev.get(queue) - res_reference) / norm(res_reference) < 1e-6)

.. testoutput:: matrixmul_example
    :hide:

    True

Most of the code above should be already familiar, with the exception of the creation of :py:class:`~reikna.linalg.MatrixMul` object.
The computation constructor takes two array-like objects, representing arrays that will participate in the computation.
After that the computation object has to be compiled.
The :py:meth:`~reikna.core.Computation.compile` method requires a :py:class:`grunnur.Device` object (or an iterable thereof), which serves as a source of data about the target API and device.


Transformations
===============

Now imagine that you want to multiply complex matrices, but real and imaginary parts of your data are kept in separate arrays.
You could create additional kernels that would join your data into arrays of complex values, but this would require additional storage and additional calls to GPU.
Transformation API allows you to connect these transformations to the core computation --- matrix multiplication --- effectively adding the code into the main computation kernel and changing its signature.

Let us change the previous example and connect transformations to it.

.. testcode:: transformation_example

    import numpy
    from numpy.linalg import norm
    from grunnur import API, Context, Queue, Array, ArrayMetadata
    from reikna.core import Type
    from reikna.linalg import MatrixMul
    from reikna.transformations import combine_complex

    context = Context.from_devices([API.any().platforms[0].devices[0]])
    queue = Queue(context.device)

    shape1 = (100, 200)
    shape2 = (200, 100)

    a_re = numpy.random.randn(*shape1).astype(numpy.float32)
    a_im = numpy.random.randn(*shape1).astype(numpy.float32)
    b_re = numpy.random.randn(*shape2).astype(numpy.float32)
    b_im = numpy.random.randn(*shape2).astype(numpy.float32)

    arrays = [Array.from_host(queue, x) for x in [a_re, a_im, b_re, b_im]]
    a_re_dev, a_im_dev, b_re_dev, b_im_dev = arrays

    a_type = ArrayMetadata(shape1, numpy.complex64)
    b_type = ArrayMetadata(shape2, numpy.complex64)
    res_dev = Array.empty(queue.device, (shape1[0], shape2[1]), dtype=numpy.complex64)

    dot = MatrixMul(a_type, b_type, out_arr_t=res_dev)
    combine_a = combine_complex(a_type)
    combine_b = combine_complex(b_type)

    dot.parameter.matrix_a.connect(
        combine_a, combine_a.output, a_re=combine_a.real, a_im=combine_a.imag)
    dot.parameter.matrix_b.connect(
        combine_b, combine_b.output, b_re=combine_b.real, b_im=combine_b.imag)

    dotc = dot.compile(queue.device)

    dotc(queue, res_dev, a_re_dev, a_im_dev, b_re_dev, b_im_dev)

    res_reference = numpy.dot(a_re + 1j * a_im, b_re + 1j * b_im)

    print(norm(res_dev.get(queue) - res_reference) / norm(res_reference) < 1e-6)

.. testoutput:: transformation_example
    :hide:

    True

We have used a pre-created transformation :py:func:`~reikna.transformations.combine_complex` from :py:mod:`reikna.transformations` for simplicity; developing a custom transformation is also possible and described in :ref:`tutorial-advanced-transformation`.
From the documentation we know that it transforms two inputs into one output; therefore we need to attach it to one of the inputs of ``dot`` (identified by its name), and provide names for two new inputs.

Names to attach to are obtained from the documentation for the particular computation; for :py:class:`~reikna.linalg.MatrixMul` these are ``out``, ``a`` and ``b``.

In the current example we have attached the transformations to both inputs.
Note that the computation has a new signature now, and the compiled ``dot`` object now works with split complex numbers.
