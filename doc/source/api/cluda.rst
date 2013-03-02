.. _api-cluda:

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

.. py:class:: Context(context, queue=None, fast_math=True, async=True, owns_context=False)

    Wraps an existing context in the CLUDA context object.

    .. note::
        If you are using ``API_CUDA``, you must keep in mind the stateful nature of CUDA calls.
        Briefly, this means that there is the context stack, and the current context on top of it.
        When the :py:meth:`create` is called, the ``PyCUDA`` context gets pushed to the stack and made current.
        When the :py:class:`Context` object goes out of scope (and ``own_context`` option was set to ``True`` in the constructor), the context is popped, and it is the user's responsibility to make sure the popped context is the correct one.
        In simple single-context programs this only means that one should avoid reference cycles involving the :py:class:`Context` object.

    :param context: a context to wrap
    :type context: :py:class:`pycuda.driver.Context` object for ``API_CUDA``, or :py:class:`pyopencl.Context` object for ``API_OCL``.
    :param queue: a queue to serialize operations to.
        If not given, a new one will be created internally.
    :type queue: :py:class:`pycuda.driver.Stream` object for ``API_CUDA``, or :py:class:`pyopencl.CommandQueue` object for ``API_OCL``.
    :param fast_math: whether to enable fast mathematical operations during compilation.
    :param async: whether to execute all operations with this context asynchronously (you would generally want to set it to ``False`` only for profiling purposes).
    :param owns_context: for ``API_CUDA``, tells whether the created object is responsible for popping the context from the CUDA context stack.
        Does not have any effect for ``API_OCL``.

    .. py:classmethod:: create(device=None, fast_math=True, async=True)

        Creates the new :py:class:`tigger.cluda.api.Context` object with its own context and queues inside.
        Intended for cases when you want to base your whole program on CLUDA.

        :param device: device to create context for, element of the list returned by :py:meth:`Platform.get_devices`.
            If not given, the device will be selected internally.
        :type device: :py:class:`pycuda.driver.Device` object for ``API_CUDA``, or :py:class:`pyopencl.Device` object for ``API_OCL``.
        :param fast_math: same as in :py:class:`Context`.
        :param async: same as in :py:class:`Context`.

    .. py:attribute:: device_params

        Instance of :py:class:`DeviceParameters` class for this context's device.

    .. py:attribute:: temp_alloc

        Instance of :py:class:`~tigger.cluda.tempalloc.TemporaryManager` which handles allocations of temporary arrays (see :py:meth:`temp_array`).

    .. py:method:: supports_dtype(dtype)

        Checks if given ``numpy`` dtype can be used in kernels compiled using this context.

    .. py:method:: allocate(size)

        Creates an untyped memory allocation object of type :py:class:`Buffer` with size ``size``.

    .. py:method:: array(shape, dtype, allocator=None)

        Creates an :py:class:`Array` on GPU with given ``shape`` and ``dtype``.
        Optionally, an ``allocator`` is a callable returning any object castable to ``int`` representing the physical address on the device (for instance, :py:class:`Buffer`).

    .. py:method:: temp_array(shape, dtype, dependencies=None)

        Creates a temporary :py:class:`Array` on GPU with given ``shape`` and ``dtype``.
        In order to reduce the memory footprint of the program, the temporary array manager will allow these arrays to overlap.
        Two arrays will not overlap, if one of them was specified in ``dependencies`` for the other one.
        For a list of values ``dependencies`` takes, see the reference entry for :py:class:`~tigger.cluda.tempalloc.TemporaryManager`.

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

        Alternatively, one might use :py:meth:`Array.get`.

    .. py:method:: copy_array(arr, dest=None, src_offset=0, dest_offset=0, size=None)

        Copies array on device.

        :param dest: the effect is the same as in :py:meth:`to_device`.
        :param src_offset: offset (in items of ``arr.dtype``) in the source array.
        :param dest_offset: offset (in items of ``arr.dtype``) in the destination array.
        :param size: how many elements of ``arr.dtype`` to copy.

    .. py:method:: synchronize()

        Forcefully synchronize the context with the main thread.

    .. py:method:: compile(template_src, render_kwds=None)

        Creates a module object from the given template.

        :param template_src: Mako template source to render
        :param render_kwds: a dictionary with additional parameters
            to be used while rendering the template.
        :returns: a :py:class:`Module` object.

    .. py:method:: compile_static(template_src, name, global_size, local_size=None, local_mem=0, render_kwds=None)

        Creates a kernel object with fixed call sizes,
        which allows to overcome some backend limitations.

        :param template_src: Mako template source to render
        :param name: name of the kernel function
        :param global_size: global size to be used
        :param local_size: local size to be used.
            If ``None``, some suitable one will be picked.
        :param local_mem: (**CUDA API only**) amount of dynamically allocated local memory to be used (in bytes).
        :param render_kwds: a dictionary with additional parameters
            to be used while rendering the template.
        :returns: a :py:class:`StaticKernel` object.

.. py:class:: Buffer

    Low-level untyped memory allocation.
    Actual class depends on the API: :py:class:`pycuda.driver.DeviceAllocation` for ``API_CUDA`` and :py:class:`pyopencl.Buffer` for ``API_OCL``.

    .. py:attribute:: size

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

        List with maximum local_size for each dimension.

    .. py:attribute:: max_num_groups

        List with maximum number of workgroups for each dimension.

    .. py:attribute:: warp_size

        Warp size (nVidia), or wavefront size (AMD), or SIMD width is supposed to be the number of threads that are executed simultaneously on the same computation unit (so you can assume that they are perfectly synchronized).

    .. py:attribute:: local_mem_banks

        Number of local (shared in CUDA) memory banks is a number of successive 32-bit words you can access without getting bank conflicts.

    .. py:attribute:: local_mem_size

        Size of the local (shared in CUDA) memory per workgroup, in bytes.

    .. py:attribute:: min_mem_coalesce_width

        Dictionary ``{word_size:elements}``, where ``elements`` is the number of elements with size ``word_size`` in global memory that allow coalesced access.

.. py:class:: Module

    .. py:attribute:: source

        Contains module source code.

    .. py:attribute:: kernel_name

        Contains :py:class:`Kernel` object for the kernel ``kernel_name``.

.. py:class:: Kernel

    .. py:method:: prepare(global_size, local_size=None, local_mem=0)

        Prepare kernel for execution with given parameters.

        :param global_size: an integer or a tuple of integers,
            specifying total number of work items to run.
        :param local_size: an integer or a tuple of integers,
            specifying the size of a single work group.
            Should have the same number of dimensions as ``global_size``.
            If ``None`` is passed, some ``local_size`` will be picked internally.
        :param local_mem: (**CUDA API only**) amount of dynamic local memory (in bytes)

    .. py:method:: prepared_call(*args)

        Execute the kernel.

    .. py:method:: __call__(*args, **kwds)

        Shortcut for successive call to :py:meth:`prepare` and :py:meth:`prepared_call`.

.. py:class:: StaticKernel

    .. py:attribute:: source

        Contains module source code.

    .. py:method:: __call__(*args)

        Execute the kernel.


Temporary Arrays
----------------

Each context contains a special allocator for arrays with data that does not have to be persistent all the time.
In many cases you only want some array to keep its contents between several kernel calls.
This can be achieved by manually allocating and deallocating such arrays every time, but it slows the program down, and you have to synchronize the queue because allocation commands are not serialized.
Therefore it is advantageous to use :py:meth:`~tigger.cluda.api.Context.temp_array` method to get such arrays.
It takes a list of dependencies as an optional parameter which gives the allocator a hint about which arrays should not use the same physical allocation.

.. py:module:: tigger.cluda.tempalloc

.. autoclass:: DynamicAllocation

.. autoclass:: TemporaryManager
    :members:

.. autoclass:: TrivialManager

.. autoclass:: ZeroOffsetManager

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

.. c:macro:: CUDA

    If defined, specifies that the kernel is being compiled for CUDA API.

.. c:macro:: CTX_FAST_MATH

    If defined, specifies that the context for which the kernel is being compiled was created with the key ``fast_math``.

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

    Modifier for the local memory argument in the device-only functions.

.. c:macro:: INLINE

    Modifier for inline functions.

.. c:function:: int get_local_id(int dim)
.. c:function:: int get_group_id(int dim)
.. c:function:: int get_global_id(int dim)
.. c:function:: int get_local_size(int dim)
.. c:function:: int get_num_groups(int dim)
.. c:function:: int get_global_size(int dim)

    Local, group and global identifiers and sizes.
    In case of CUDA mimic the behavior of corresponding OpenCL functions.

.. c:macro:: VIRTUAL_SKIP_THREADS

    This macro should start any kernel compiled with :py:meth:`~tigger.cluda.api.Context.compile_static`.
    It skips all the empty threads resulting from fitting call parameters into backend limitations.

.. c:function:: int virtual_local_id(int dim)
.. c:function:: int virtual_group_id(int dim)
.. c:function:: int virtual_global_id(int dim)
.. c:function:: int virtual_local_size(int dim)
.. c:function:: int virtual_num_groups(int dim)
.. c:function:: int virtual_global_size(int dim)

    Only available in :py:class:`~tigger.cluda.api.StaticKernel` objects obtained from :py:meth:`~tigger.cluda.api.Context.compile_static`.
    Since its dimensions can differ from actual call dimensions, these functions have to be used.

.. c:function:: int virtual_global_flat_id(int dim)
.. c:function:: int virtual_global_flat_size(int dim)

    Only available in :py:class:`~tigger.cluda.api.StaticKernel` objects obtained from :py:meth:`~tigger.cluda.api.Context.compile_static`.
    useful for addressing input and output arrays.

Datatype tools
--------------

This module contains various convenience functions which operate with :py:class:`numpy.dtype` objects.

.. automodule:: tigger.cluda.dtypes
    :members:
