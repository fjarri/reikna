.. _api-cluda:

CLUDA layer
===========

CLUDA is the foundation of ``reikna``.
It provides the unified access to basic features of ``CUDA`` and ``OpenCL``, such as memory operations, compilation and so on.
It can also be used by itself, if you want to write GPU API-independent programs and happen to only need a small subset of GPU API.
The terminology is borrowed from ``OpenCL``, since it is a more general API.

.. automodule:: reikna.cluda
   :members:
   :imported-members:


API module
----------

Modules for all APIs have the same generalized interface.
It is referred here (and references from other parts of this documentation) as :py:mod:`reikna.cluda.api`.

.. automodule:: reikna.cluda.api
    :members:
    :special-members: __call__


Temporary Arrays
----------------

Each :py:class:`~reikna.cluda.api.Thread` contains a special allocator for arrays with data that does not have to be persistent all the time.
In many cases you only want some array to keep its contents between several kernel calls.
This can be achieved by manually allocating and deallocating such arrays every time, but it slows the program down, and you have to synchronize the queue because allocation commands are not serialized.
Therefore it is advantageous to use :py:meth:`~reikna.cluda.api.Thread.temp_array` method to get such arrays.
It takes a list of dependencies as an optional parameter which gives the allocator a hint about which arrays should not use the same physical allocation.

.. py:module:: reikna.cluda.tempalloc

.. autoclass:: TemporaryManager
    :members:

.. autoclass:: TrivialManager

.. autoclass:: ZeroOffsetManager


Function modules
----------------

.. automodule :: reikna.cluda.functions
    :members:


.. _cluda-kernel-toolbox:

Kernel toolbox
--------------

.. py:module:: reikna.cluda.kernel

The stuff available for the kernel passed for compilation consists of two parts.

First, there are several objects available at the template rendering stage, namely ``numpy``, :py:mod:`reikna.cluda.dtypes` (as ``dtypes``), and :py:mod:`reikna.helpers` (as ``helpers``).

Second, there is a set of macros attached to any kernel depending on the API it is being compiled for:

.. c:macro:: CUDA

    If defined, specifies that the kernel is being compiled for CUDA API.

.. c:macro:: COMPILE_FAST_MATH

    If defined, specifies that the compilation for this kernel was requested with ``fast_math == True``.

.. c:macro:: LOCAL_BARRIER

    Synchronizes threads inside a block.

.. c:macro:: WITHIN_KERNEL

    Modifier for a device-only function declaration.

.. c:macro:: KERNEL

    Modifier for a kernel function declaration.

.. c:macro:: GLOBAL_MEM

    Modifier for a global memory pointer argument.

.. c:macro:: LOCAL_MEM

    Modifier for a statically allocated local memory variable.

.. c:macro:: LOCAL_MEM_DYNAMIC

    Modifier for a dynamically allocated local memory variable.

.. c:macro:: LOCAL_MEM_ARG

    Modifier for a local memory argument in device-only functions.

.. c:macro:: CONSTANT_MEM

    Modifier for a statically allocated constant memory variable.

.. c:macro:: CONSTANT_MEM_ARG

    Modifier for a constant memory argument in device-only functions.

.. c:macro:: INLINE

    Modifier for inline functions.

.. c:macro:: SIZE_T

    The type of local/global IDs and sizes.
    Equal to ``unsigned int`` for CUDA, and ``size_t`` for OpenCL
    (which can be 32- or 64-bit unsigned integer, depending on the device).

.. FIXME: techincally, it should be unsigned int here, but Sphinx gives warnings for 'unsigned'
.. c:function:: SIZE_T get_local_id(int dim)
.. c:function:: SIZE_T get_group_id(int dim)
.. c:function:: SIZE_T get_global_id(int dim)
.. c:function:: SIZE_T get_local_size(int dim)
.. c:function:: SIZE_T get_num_groups(int dim)
.. c:function:: SIZE_T get_global_size(int dim)

    Local, group and global identifiers and sizes.
    In case of CUDA mimic the behavior of corresponding OpenCL functions.

.. c:macro:: VSIZE_T

    The type of local/global IDs in the virtual grid.
    It is separate from :c:macro:`SIZE_T` because the former is intended to be equivalent to
    what the backend is using, while ``VSIZE_T`` is a separate type and can be made larger
    than ``SIZE_T`` in the future if necessary.

.. c:macro:: ALIGN(int)

    Used to specify an explicit alignment (in bytes) for fields in structures, as

    ::

        typedef struct {
            char ALIGN(4) a;
            int b;
        } MY_STRUCT;

.. c:macro:: VIRTUAL_SKIP_THREADS

    This macro should start any kernel compiled with :py:meth:`~reikna.cluda.api.Thread.compile_static`.
    It skips all the empty threads resulting from fitting call parameters into backend limitations.

.. FIXME: techincally, it should be unsigned int here, but Sphinx gives warnings for 'unsigned'
.. c:function:: VSIZE_T virtual_local_id(int dim)
.. c:function:: VSIZE_T virtual_group_id(int dim)
.. c:function:: VSIZE_T virtual_global_id(int dim)
.. c:function:: VSIZE_T virtual_local_size(int dim)
.. c:function:: VSIZE_T virtual_num_groups(int dim)
.. c:function:: VSIZE_T virtual_global_size(int dim)
.. c:function:: VSIZE_T virtual_global_flat_id()
.. c:function:: VSIZE_T virtual_global_flat_size()

    Only available in :py:class:`~reikna.cluda.api.StaticKernel` objects obtained from :py:meth:`~reikna.cluda.api.Thread.compile_static`.
    Since its dimensions can differ from actual call dimensions, these functions have to be used.


Datatype tools
--------------

This module contains various convenience functions which operate with ``numpy.dtype`` objects.

.. automodule:: reikna.cluda.dtypes
    :members:
