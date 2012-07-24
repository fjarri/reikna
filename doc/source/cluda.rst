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

.. py:attribute:: API_ID

    Identifier of this API.

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

        :param device: device to create context for.
            If not given, the device will be selected internally.
        :type device: :py:class:`pycuda.driver.Device` object for ``API_CUDA``, or :py:class:`pyopencl.Device` object for ``API_OCL``.
        :param fast_math: same as in :py:class:`Context`.
        :param async: same as in :py:class:`Context`.

    .. py:method:: supports_dtype(dtype)

        Checks if given ``numpy`` dtype can be used in kernels compiled using this context.

    .. py:method:: allocate(shape, dtype)

        Create an array on GPU with given ``shape`` and ``dtype``.
        For ``API_CUDA``, returns ``pycuda.gpuarray.GPUArray`` object, for ``API_OCL`` returns ``pyopencl.array.Array``.

    .. py:method:: empty_like(arr)

        Allocates an array on GPU with the same shape and dtype as ``arr``.

    .. py:method:: to_device(arr)

        Copies an array to the device memory.
