.. _cluda-reference:

CLUDA layer
===========

CLUDA is the foundation of the tigger.
It provides unified access to basic features of CUDA and OpenCL, such as memory operations, compilation and so on.
It can also be used by itself, if you want to write GPU API-independent programs and happen to need the small subset of GPU API.
The terminology is borrowed from OpenCL, since it is a more general API.

Root level interface
--------------------

This module contains functions for API discovery.

.. automodule:: tigger.cluda
   :members:

API module
----------

Modules for all APIs have the same generalized interface.
It is referred here (and references from other parts of this documentation) as :py:mod:`tigger.cluda.api`.

.. py:module:: tigger.cluda.api

.. py:data:: API_ID

    Identifier of this API.

.. py:function:: get_platforms()

    Returns a list of platform objects.
    The methods and attributes available are described in the reference entry for :py:class:`Platform`.
    In case of ``API_OCL`` returned objects are actually instances of :py:class:`pyopencl.Platform`.

.. py:class:: Platform

    .. py:attribute:: name

        String with platform name.

    .. py:attribute:: vendor

        String with platform vendor.

    .. py:attribute:: version

        String with platform version.

    .. py:method:: get_devices()

        Returns a list of device objects from the platform.

.. py:class:: Context(context, stream=None, fast_math=True, async=True)

    Wraps existing context in the CLUDA context object.

    :param context: a context to wrap
    :type context: :py:class:`pycuda.driver.Context` object for ``API_CUDA``, or :py:class:`pyopencl.Context` object for ``API_OCL``.
    :param stream: a stream to serialize operations to.
        If not given, a new one will be created internally.
    :type stream: :py:class:`pycuda.driver.Stream` object for ``API_CUDA``, or :py:class:`pyopencl.CommandQueue` object for ``API_OCL``.
    :param fast_math: whether to enable fast mathematical operations during compilation.
    :param async: whether to execute all operations with this context asynchronously (you would generally want to set it to ``False`` only for profiling purposes).

    .. py:classmethod:: create(device=None, fast_math=True, async=True)

        Creates the new :py:class:`tigger.cluda.api.Context` object with its own context and stream inside.
        Intended for cases when you want to base your whole program on CLUDA.

        :param device: device to create context for, element of the list returned by :py:meth:`Platform.get_devices`.
            If not given, the device will be selected internally.
        :type device: :py:class:`pycuda.driver.Device` object for ``API_CUDA``, or :py:class:`pyopencl.Device` object for ``API_OCL``.
        :param fast_math: same as in :py:class:`Context`.
        :param async: same as in :py:class:`Context`.

    .. py:attribute:: device_params

        Instance of :py:class:`DeviceParameters` class for this context's device.

    .. py:method:: supports_dtype(dtype)

        Checks if given ``numpy`` dtype can be used in kernels compiled using this context.

    .. py:method:: allocate(shape, dtype)

        Creates an :py:class:`Array` on GPU with given ``shape`` and ``dtype``.

    .. py:method:: empty_like(arr)

        Allocates an array on GPU with the same shape and dtype as ``arr``.

    .. py:method:: to_device(arr, dest=None)

        Copies an array to the device memory.
        If ``dest`` is specified, it is used as the destination, and the method returns ``None``.
        Otherwise the destination array is created internally and returned from the method.

    .. py:method:: from_device(arr, dest=None, async=False)

        Transfers the contents of ``arr`` to a :py:class:`numpy.ndarray` object.
        The effect of ``dest`` parameter is the same as in :py:meth:`to_device`.
        If ``async`` is ``True``, the transfer is asynchronous (the context-wide asynchronisity setting does not apply here).

        Alternatively, one might use :py:meth:`Array.get()`.

    .. py:method:: copy_array(arr, dest=None, src_offset=0, dest_offset=0, size=None)

        Copies array on device.

        :param dest: the effect is the same as in :py:meth:`to_device`.
        :param src_offset: offset (in items of ``arr.dtype``) in the source array.
        :param dest_offset: offset (in items of ``arr.dtype``) in the destination array.
        :param size: how many elements of ``arr.dtype`` to copy.

    .. py:method:: synchronize()

        Forcefully synchronize the context with the main thread.

    .. py:method:: compile(template_src, **kwds)

        Compiles ``Mako`` template source with ``kwds`` passed to the ``render()`` function and returns :py:class:`Module` object.
        In addition to ``kwds``, some pre-defined keywords, and defines from API-specific prelude are available inside the kernel.
        See :ref:`Kernel toolbox <cluda-kernel-toolbox>` for details.

    .. py:method:: release()

        Release and invalidate the context.
        This happens automatically on object deletion, so call it only if you want to release resources earlier than object lifecycle takes care of that.

        Does not have any effect if the :py:class:`Context` was created as a wrapper for the existing context.

.. py:class:: Array

    Actual array class is different depending on the API: :py:class:`pycuda.gpuarray.GPUArray` for ``API_CUDA`` and :py:class:`pyopencl.array.Array` for ``API_OCL``.
    This is an interface they both provide.

    .. py:attribute:: shape

    .. py:attribute:: dtype

    .. py:method:: get()

        Returns :py:class:`numpy.ndarray` with the contents of the array.
        Synchronizes the context.

.. py:class:: DeviceParameters

    An assembly of device parameters necessary for optimizations.

    .. py:attribute:: max_work_group_size

        Maximum block size for kernels.

    .. py:attribute:: max_work_item_sizes

        3-element list with maximum block dimensions.

    .. py:attribute:: max_grid_dims

        2-element list with maximum grid dimensions.

    .. py:attribute:: warp_size

        Warp size (nVidia), or wavefront size (AMD), or SIMD width is supposed to be the number of threads that are executed simultaneously on the same computation unit (so you can assume that they are perfectly synchronized).

    .. py:attribute:: smem_banks

        Shared (local for AMD) memory banks is a number of successive 32-bit words you can access without getting bank conflicts.

.. py:class:: Module

    .. py:attribute:: kernel_name

        Contains :py:class:`Kernel` object for the kernel ``kernel_name``.

.. py:class:: Kernel

    .. py:method:: prepare(block=(1, 1, 1), grid=(1, 1), shared=0)

        Prepare kernel for execution with given parameters.

    .. py:method:: prepared_call(*args)

        Execute kernel.

    .. py:method:: __call__(*args, **kwds)

        Shortcut for successive call to :py:meth:`prepare` and :py:meth:`prepared_call`.


.. _cluda-kernel-toolbox:

Kernel toolbox
--------------

The stuff available for the kernel passed for compilation consists of two parts.

First, there are several objects available at the template rendering stage, namely ``numpy``, :py:mod:`tigger.cluda.dtypes` (as ``dtypes``) and a :py:class:`FuncCollector` instance named ``func``, which is used to compensate for the lack of complex number operations in OpenCL, and the lack of C++ synthax which would allow one to write them.
Its methods can be treated as if they return the name of the function necessary to operate on given dtypes.
Available methods are:

.. py:module:: tigger.cluda.kernel

.. py:class:: FuncCollector

    .. py:method:: mul(dtype1, dtype2, out=None)

        Returns the name of the function that multiplies values of types ``dtype1`` and ``dtype2``.
        If ``out`` is given, it will be set as a return type for this function.

    .. py:method:: div(dtype1, dtype2, out=None)

        Returns the name of the function that divides values of ``dtype1`` and ``dtype2``.
        If ``out`` is given, it will be set as a return type for this function.

    .. py:method:: cast(out_dtype, in_dtype)

        Returns the name of the function that casts values of ``in_dtype`` to ``out_dtype``.

Second, there is a set of macros attached to any kernel depending on the API it is being compiled for:

.. c:macro:: LOCAL_BARRIER

    Synchronizes threads inside a block.

.. c:macro:: WITHIN_KERNEL

    Modifier for a device-only function declaration.

.. c:macro:: KERNEL

    Modifier for the kernel function declaration.

.. c:macro:: GLOBAL_MEM

    Modifier for the global memory pointer argument.

.. c:macro:: LOCAL_MEM

    Modifier for the statically allocated local memory variable.

.. c:macro:: LOCAL_MEM_DYNAMIC

    Modifier for the dynamically allocated local memory variable.

.. c:macro:: LOCAL_MEM_ARG

    Modifier for the local memory argument to the device-only functions.

.. c:macro:: INLINE

    Modifier for inline functions.

.. c:macro:: LID_0
.. c:macro:: LID_1
.. c:macro:: LID_2

    Thread identifiers in a block for three dimensions.

.. c:macro:: GID_0
.. c:macro:: GID_1
.. c:macro:: GID_2

    Block identifiers in a grid for three dimensions.

.. c:macro:: LSIZE_0
.. c:macro:: LSIZE_1
.. c:macro:: LSIZE_2

    Block sizes for three dimensions.


Datatype tools
--------------

This module contains various convenience functions which operate with :py:class:`numpy.dtype` objects.

.. automodule:: tigger.cluda.dtypes
    :members:
