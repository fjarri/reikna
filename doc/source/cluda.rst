.. _cluda-reference:

CLUDA layer
===========

CLUDA is the foundation of the tigger.
It provides unified access to basic features of CUDA and OpenCL, such as memory operations, compilation and so on.
It can also be used by itself, if you want to write GPU API-independent programs and happen to need the small subset of GPU API.

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

    .. py:attribute:: max_block_size

        Maximum block size for kernels.

    .. py:attribute:: max_block_dims

        3-element list with maximum block dimensions.

    .. py:attribute:: max_grid_dims

        2-element list with maximum grid dimensions.

    .. py:attribute:: warp_size

        Warp size (nVidia), or wavefront size (AMD), or SIMD width is supposed to be the number of threads that are executed simultaneously on the same computation unit (so you can assume that they are perfectly synchronized).

    .. py:attribute:: smem_banks

        Shared (local for AMD) memory banks is a number of successive 32-bit words you can access without getting bank conflicts.
