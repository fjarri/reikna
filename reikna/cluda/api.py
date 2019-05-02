"""
.. py:class:: Buffer

    Low-level untyped memory allocation.
    Actual class depends on the API: ``pycuda.driver.DeviceAllocation`` for ``CUDA``
    and ``pyopencl.Buffer`` for ``OpenCL``.

    .. py:attribute:: size

.. py:class:: Array

    A superclass of the corresponding API's native array
    (``pycuda.gpuarray.GPUArray`` for ``CUDA`` and ``pyopencl.array.Array`` for ``OpenCL``),
    with some additional functionality.

    .. py:attribute:: shape

    .. py:attribute:: dtype

    .. py:attribute:: strides

    .. py:attribute:: offset

        The start of the array data in the memory buffer (in bytes).

    .. py:attribute:: base_data

        The memory buffer where the array is located.

    .. py:attribute:: nbytes

        The total size of the array data plus the offset (in bytes).

    .. py:method:: roll(shift, axis=-1)

        Cyclically shifts elements of ``array`` **inplace**
        by ``shift`` positions to the right along ``axis``.
        ``shift`` can be negative (in which case the elements are shifted to the left).
        Elements that are shifted beyond the last position are re-introduced at the first
        (and vice versa).

        Works equivalently to ``numpy.roll`` (except ``axis=None`` is not supported).

    .. py:method:: get()

        Returns ``numpy.ndarray`` with the contents of the array.
        Synchronizes the parent :py:class:`~reikna.cluda.api.Thread`.

    .. py:attribute:: thread

        The :py:class:`Thread` object for which the array was created.

.. py:class:: DeviceParameters(device)

    An assembly of device parameters necessary for optimizations.

    .. py:attribute:: api_id

        Identifier of the API this device belongs to.

    .. py:attribute:: max_work_group_size

        Maximum block size for kernels.

    .. py:attribute:: max_work_item_sizes

        List with maximum local_size for each dimension.

    .. py:attribute:: max_num_groups

        List with maximum number of workgroups for each dimension.

    .. py:attribute:: warp_size

        Warp size (nVidia), or wavefront size (AMD), or SIMD width is supposed to be
        the number of threads that are executed simultaneously on the same computation unit
        (so you can assume that they are perfectly synchronized).

    .. py:attribute:: local_mem_banks

        Number of local (shared in CUDA) memory banks is a number of successive 32-bit words
        you can access without getting bank conflicts.

    .. py:attribute:: local_mem_size

        Size of the local (shared in CUDA) memory per workgroup, in bytes.

    .. py:attribute:: min_mem_coalesce_width

        Dictionary ``{word_size:elements}``, where ``elements`` is the number of elements
        with size ``word_size`` in global memory that allow coalesced access.

    .. py:attribute:: compute_units

        The value of ``MULTIPROCESSOR_COUNT`` in CUDA and ``MAX_COMPUTE_UNITS`` in OpenCL.

    .. py:method:: supports_dtype(self, dtype)

        Checks if given ``numpy`` dtype can be used in kernels compiled using this thread.

.. py:class:: Platform

    A vendor-specific implementation of the GPGPU API.

    .. py:attribute:: name

        Platform name.

    .. py:attribute:: vendor

        Vendor name.

    .. py:attribute:: version

        Platform version.

    .. py:method:: get_devices()

        Returns a list of device objects available in the platform.
"""

from __future__ import print_function
from logging import error
import weakref
import sys

from reikna.cluda import find_devices
from reikna.cluda.api_discovery import cuda_id
from reikna.helpers import product, wrap_in_tuple
from reikna.cluda.kernel import render_prelude, render_template_source
from reikna.cluda.vsize import VirtualSizes
from reikna.cluda.tempalloc import ZeroOffsetManager

_input = input if sys.version_info[0] >= 3 else raw_input


def get_id():
    """
    Returns the identifier of this API.
    """
    raise NotImplementedError()


def get_platforms():
    """
    Returns a list of available :py:class:`Platform` objects.
    In case of ``OpenCL`` returned objects are actually instances of ``pyopencl.Platform``.
    """
    raise NotImplementedError()


class Thread:
    """
    Wraps an existing context in the CLUDA thread object.

    :param cqd: a ``Context``, ``Device`` or ``Stream``/``CommandQueue`` object to base on.
        If a context is passed, a new stream/queue will be created internally.
    :param async_: whether to execute all operations with this thread asynchronously
        (you would generally want to set it to ``False`` only for profiling purposes).

    .. note::
        If you are using ``CUDA`` API, you must keep in mind the stateful nature of CUDA calls.
        Briefly, this means that there is the context stack, and the current context on top of it.
        When the :py:meth:`create` is called, the ``PyCUDA`` context gets pushed to the stack
        and made current.
        When the thread object goes out of scope (and the thread object owns it),
        the context is popped, and it is the user's responsibility to make sure
        the popped context is the correct one.
        In simple single-context programs this only means that one should avoid reference cycles
        involving the thread object.

    .. warning::

        Do not pass one ``Stream``/``CommandQueue`` object to several ``Thread`` objects.

    .. py:attribute:: api

        Module object representing the CLUDA API corresponding to this ``Thread``.

    .. py:attribute:: device_params

        Instance of :py:class:`DeviceParameters` class for this thread's device.

    .. py:attribute:: temp_alloc

        Instance of :py:class:`~reikna.cluda.tempalloc.TemporaryManager`
        which handles allocations of temporary arrays (see :py:meth:`temp_array`).
    """

    @classmethod
    def create(cls, interactive=False, device_filters=None, **thread_kwds):
        """
        Creates a new ``Thread`` object with its own context and queue inside.
        Intended for cases when you want to base your whole program on CLUDA.

        :param interactive: ask a user to choose a platform and a device from the ones found.
            If there is only one platform/device available, they will be chosen automatically.
        :param device_filters: keywords to filter devices
            (see the keywords for :py:func:`~reikna.cluda.find_devices`).
        :param thread_kwds: keywords to pass to :py:class:`Thread` constructor.
        """

        if device_filters is None:
            device_filters = {}
        devices = find_devices(cls.api, **device_filters)

        platforms = cls.api.get_platforms()

        if interactive:
            pnums = sorted(devices.keys())
            if len(pnums) == 1:
                selected_pnum = pnums[0]
                print("Platform:", platforms[0].name)
            else:
                print("Platforms:")
                default_pnum = pnums[0]
                for pnum in pnums:
                    print("[{pnum}]: {pname}".format(pnum=pnum, pname=platforms[pnum].name))
                print(
                    "Choose the platform [{default_pnum}]:".format(default_pnum=default_pnum),
                    end='')
                selected_pnum = _input()
                if selected_pnum == '':
                    selected_pnum = default_pnum
                else:
                    selected_pnum = int(selected_pnum)

            platform = platforms[selected_pnum]
            dnums = sorted(devices[selected_pnum])
            if len(dnums) == 1:
                selected_dnum = dnums[0]
                print("Device:", platform.get_devices()[selected_dnum].name)
            else:
                print("Devices:")
                default_dnum = dnums[0]
                for dnum in dnums:
                    print("[{dnum}]: {dname}".format(
                        dnum=dnum, dname=platform.get_devices()[dnum].name))
                print(
                    "Choose the device [{default_dnum}]:".format(default_dnum=default_dnum),
                    end='')
                selected_dnum = _input()
                if selected_dnum == '':
                    selected_dnum = default_dnum
                else:
                    selected_dnum = int(selected_dnum)

        else:
            selected_pnum = sorted(devices.keys())[0]
            selected_dnum = devices[selected_pnum][0]

        if thread_kwds is None:
            thread_kwds = {}
        return cls(platforms[selected_pnum].get_devices()[selected_dnum], **thread_kwds)

    def __init__(self, cqd, async_=True, temp_alloc=None):

        self._released = False
        self._async = async_

        # Make the fields initialized even in case _prcess_cqd() raises an exception.
        self._context = None
        self._queue = None
        self._device = None
        self._owns_context = False
        self._context, self._queue, self._device, self._owns_context = self._process_cqd(cqd)

        self.device_params = self.api.DeviceParameters(self._device)

        temp_alloc_params = dict(
            cls=ZeroOffsetManager, pack_on_alloc=False, pack_on_free=False)
        if temp_alloc is not None:
            temp_alloc_params.update(temp_alloc)

        self.temp_alloc = temp_alloc_params['cls'](weakref.proxy(self),
            pack_on_alloc=temp_alloc_params['pack_on_alloc'],
            pack_on_free=temp_alloc_params['pack_on_free'])

        self._computation_cache = dict()

    def get_cached_computation(self, cls, *args, **kwds):
        """
        Returns a compiled computation ``cls`` initialized with ``args`` and ``kwds``.
        The results are cached, so any computation with the same arguments will only be
        initialized and compiled once.

        .. note::

            All of ``args`` and ``kwds`` must be hashable! If any of those are arrays,
            they can be passed through ``Type.from_value()``.
        """

        # TODO: we could make given numpy arrays and GPU arrays hashable
        # by calling ``Type.from_value()`` on them automatically,
        # but that would require importing the ``Array`` type, creating a circular dependency.

        hashable_kwds = tuple((key, kwds[key]) for key in sorted(kwds))
        key = (cls, args, hashable_kwds)
        if key in self._computation_cache:
            return self._computation_cache[key]
        else:
            comp = cls(*args, **kwds)
            compiled_comp = comp.compile(self)
            self._computation_cache[key] = compiled_comp
            return compiled_comp

    def allocate(self, size):
        """
        Creates an untyped memory allocation object of type :py:class:`Buffer` with size ``size``.
        """
        raise NotImplementedError()

    def array(
            self, shape, dtype, strides=None, offset=0, nbytes=None,
            base=None, base_data=None, allocator=None):
        """
        Creates an :py:class:`Array` on GPU with given ``shape``, ``dtype``,
        ``strides`` and ``offset``.

        If ``nbytes`` is ``None``, the size of the allocated memory buffer is chosen
        to be the minimum one to fit all the elements of the array, based on ``shape``,
        ``dtype`` and ``strides`` (if provided).
        If ``offset`` is not 0, an *additional* ``offset`` bytes is added at the beginning
        of the buffer.

        .. note::

            Reikna computations (including the template functions
            :py:meth:`~reikna.core.transformation.KernelParameter.load_idx`,
            :py:meth:`~reikna.core.transformation.KernelParameter.store_idx` etc),
            high-level PyCUDA/PyOpenCL functions and PyCUDA kernels
            take ``offset`` into account automatically and address arrays starting
            from the position of the actual data.
            Reikna kernels (created with :py:meth:`~reikna.cluda.api.Thread.compile`
            and :py:meth:`~reikna.cluda.api.Thread.compile_static`) and PyOpenCL kernels
            receive **base addresses** of arrays, and thus have to add offsets manually.

        If ``base``, ``base_data`` and ``nbytes`` are ``None``,
        the total allocated size will be the minimum size required for the array data
        (based on ``shape`` and ``strides``) plus ``offset``.

        If ``base`` and ``base_data`` are ``None``, but ``nbytes`` is not,
        ``nbytes`` bytes will be allocated for the array (this includes the offset).

        If ``base_data`` (a memory buffer) is not ``None``,
        it will be used as the underlying buffer for the array,
        with the actual data starting at the ``offset`` bytes from the beginning of ``base_data``.
        No size checking to make sure the array and the offset fit it will be performed.

        If ``base`` (an :py:class:`Array` object) is not ``None``,
        its buffer is used as the underlying buffer for the array,
        with the actual data starting at the ``offset`` bytes from the beginning
        of ``base.base_data``.
        ``base_data`` will be ignored.

        Optionally, an ``allocator`` is a callable returning any object castable to ``int``
        representing the physical address on the device (for instance, :py:class:`Buffer`).
        """
        raise NotImplementedError()

    def temp_array(self, shape, dtype, strides=None, offset=0, nbytes=None, dependencies=None):
        """
        Creates an :py:class:`Array` on GPU with given ``shape``, ``dtype``,
        ``strides``, ``offset`` and ``nbytes`` (see :py:meth:`array` for details).
        In order to reduce the memory footprint of the program, the temporary array manager
        will allow these arrays to overlap.
        Two arrays will not overlap, if one of them was specified in ``dependencies``
        for the other one.
        For a list of values ``dependencies`` takes, see the reference entry for
        :py:class:`~reikna.cluda.tempalloc.TemporaryManager`.
        """
        return self.temp_alloc.array(
            shape, dtype, strides=strides, offset=offset, nbytes=nbytes,
            dependencies=dependencies)

    def empty_like(self, arr):
        """
        Allocates an array on GPU with the same attributes (``shape``, ``dtype``, ``strides``,
        ``offset`` and ``nbytes``) as ``arr``.

        .. warning::

            Note that ``pycuda.GPUArray`` objects do not have the ``offset`` attribute.
        """
        if hasattr(arr, 'allocator'):
            allocator = arr.allocator
        else:
            allocator = None
        if hasattr(arr, 'strides'):
            strides = arr.strides
        else:
            strides = None
        if hasattr(arr, 'offset'):
            offset = arr.offset
        else:
            offset = 0
        if hasattr(arr, 'nbytes'):
            nbytes = arr.nbytes
        else:
            nbytes = None
        return self.array(
            arr.shape, arr.dtype, strides=strides, offset=offset, nbytes=nbytes,
            allocator=allocator)

    def to_device(self, arr, dest=None):
        """
        Copies an array to the device memory.
        If ``dest`` is specified, it is used as the destination, and the method returns ``None``.
        Otherwise the destination array is created internally and returned from the method.
        """
        if dest is None:
            arr_device = self.empty_like(arr)
        else:
            arr_device = dest

        self._copy_array(arr_device, arr)
        self._synchronize()

        if dest is None:
            return arr_device

    def from_device(self, arr, dest=None, async_=False):
        """
        Transfers the contents of ``arr`` to a ``numpy.ndarray`` object.
        The effect of ``dest`` parameter is the same as in :py:meth:`to_device`.
        If ``async_`` is ``True``, the transfer is asynchronous
        (the thread-wide asynchronisity setting does not apply here).

        Alternatively, one can use :py:meth:`Array.get`.
        """
        raise NotImplementedError()

    def copy_array(self, arr, dest=None, src_offset=0, dest_offset=0, size=None):
        """
        Copies array on device.

        :param dest: the effect is the same as in :py:meth:`to_device`.
        :param src_offset: offset (in items of ``arr.dtype``) in the source array.
        :param dest_offset: offset (in items of ``arr.dtype``) in the destination array.
        :param size: how many elements of ``arr.dtype`` to copy.
        """

        if dest is None:
            arr_device = self.empty_like(arr)
        else:
            arr_device = dest

        itemsize = arr.dtype.itemsize
        nbytes = arr.nbytes if size is None else itemsize * size
        src_offset *= itemsize
        dest_offset *= itemsize

        self._copy_array_buffer(arr_device, arr,
            nbytes, src_offset=src_offset, dest_offset=dest_offset)
        self._synchronize()

        if dest is None:
            return arr_device

    def synchronize(self):
        """
        Forcefully synchronize this thread with the main program.
        """
        raise NotImplementedError()

    def _synchronize(self):
        if not self._async:
            self.synchronize()

    def _create_program(self, src, fast_math=False, compiler_options=None, keep=False):
        try:
            program = self._compile(
                src, fast_math=fast_math, compiler_options=compiler_options, keep=keep)
        except:
            listing = "\n".join([str(i+1) + ":" + l for i, l in enumerate(src.split('\n'))])
            error("Failed to compile:\n" + listing)
            raise
        return program

    def compile(
            self, template_src, render_args=None, render_kwds=None, fast_math=False,
            compiler_options=None, constant_arrays=None, keep=False):
        """
        Creates a module object from the given template.

        :param template_src: Mako template source to render
        :param render_args: an iterable with positional arguments to pass to the template.
        :param render_kwds: a dictionary with keyword parameters to pass to the template.
        :param fast_math: whether to enable fast mathematical operations during compilation.
        :param compiler_options: a list of strings to be passed to the compiler as arguments.
        :param constant_arrays: (**CUDA only**) a dictionary ``{name: metadata}``
            of constant memory arrays to be declared in the compiled program.
            ``metadata`` can be either an array-like object (possessing ``shape`` and ``dtype``
            attributes), or a pair ``(shape, dtype)``.
        :param keep: if `True`, preserve the source file being compiled
            and the accompanying binaries (if any).
            With PyCUDA backend, it is used as the ``keep`` option when creating ``SourceModule``.
            With PyOpenCL backend, it is used as the ``cache_dir`` option for ``Program.build()``
            (and, additionally, the kernel source itself is put there).
        :returns: a :py:class:`Program` object.
        """
        src = render_template_source(
            template_src, render_args=render_args, render_kwds=render_kwds)
        return Program(
            self, src, fast_math=fast_math, compiler_options=compiler_options,
            constant_arrays=constant_arrays, keep=keep)

    def compile_static(self, template_src, name, global_size,
            local_size=None, render_args=None, render_kwds=None, fast_math=False,
            compiler_options=None, constant_arrays=None, keep=False):
        """
        Creates a kernel object with fixed call sizes,
        which allows to overcome some backend limitations.
        Global and local sizes can have any length, providing that
        ``len(global_size) >= len(local_size)``, and the total number of work items and work groups
        is less than the corresponding total number available for the device.
        In order to get IDs and sizes in such kernels, virtual size functions have to be used
        (see :c:macro:`VIRTUAL_SKIP_THREADS` and others for details).

        :param template_src: Mako template or a template source to render
        :param name: name of the kernel function
        :param global_size: global size to be used, in **row-major** order.
        :param local_size: local size to be used, in **row-major** order.
            If ``None``, some suitable one will be picked.
        :param local_mem: (**CUDA API only**) amount of dynamically allocated local memory
            to be used (in bytes).

        The rest of the keyword parameters are the same as for :py:meth:`compile`.

        :returns: a :py:class:`StaticKernel` object.
        """
        return StaticKernel(self, template_src, name, global_size,
            local_size=local_size, render_args=render_args, render_kwds=render_kwds,
            fast_math=fast_math, compiler_options=compiler_options,
            constant_arrays=constant_arrays, keep=keep)

    def _release_specific(self):
        """
        Overridden by a specific ``Thread`` if it needs to do something before finalizing.
        """
        pass

    def release(self):
        """
        Forcefully free critical resources (rendering the object unusable).
        In most cases you can rely on the garbage collector taking care of things.
        Calling this method explicitly may be necessary in case of CUDA API
        when you want to make sure the context got popped.
        """
        # Note that we rely on reference counting here, which is not available
        # in some Python implementations.
        # But since PyCUDA and PyOpenCL only work in CPython anyway,
        # and I cannot see the way to make Thread release both GC-only-compatible
        # and auto-releasable when not in use (crucial for CUDA because of its
        # stupid stateful contexts), I'll leave at is it is for now.

        self._computation_cache.clear()

        try:
            released = self._released
        except AttributeError:
            # Can happen if __init__() was not even completed
            return

        if not released:
            self._release_specific()
            del self._device
            del self._queue
            del self._context
            self._released = True


def normalize_constant_arrays(constant_arrays):
    if constant_arrays is None:
        return {}

    normalized = {}
    for name, metadata in constant_arrays.items():
        if hasattr(metadata, 'shape') and hasattr(metadata, 'dtype'):
            shape, dtype = metadata.shape, metadata.dtype
        else:
            shape, dtype = metadata
            shape = wrap_in_tuple(shape)
        length = product(shape)
        normalized[name] = (length, dtype)

    return normalized


class Program(object):
    """
    An object with compiled GPU code.

    .. py:attribute:: source

        Contains module source code.

    .. py:attribute:: kernel_name

        Contains :py:class:`Kernel` object for the kernel ``kernel_name``.
    """

    def __init__(
            self, thr, src, static=False, fast_math=False, compiler_options=None,
            constant_arrays=None, keep=False):
        """__init__()""" # hide the signature from Sphinx

        self._thr = thr
        self._static = static

        constant_arrays = normalize_constant_arrays(constant_arrays)
        if len(constant_arrays) > 0 and self._thr.api.get_id() != cuda_id():
            raise ValueError("Compile-time constant arrays are only supported on the CUDA platform")

        prelude = render_prelude(self._thr, fast_math=fast_math, constant_arrays=constant_arrays)

        # Casting source code to ASCII explicitly
        # New versions of Mako produce Unicode output by default,
        # and it makes the compiler unhappy
        self.source = str(prelude + src)

        self._constant_arrays = constant_arrays
        self._program = thr._create_program(
            self.source, fast_math=fast_math, compiler_options=compiler_options, keep=keep)

    def __getattr__(self, name):
        return self._thr.api.Kernel(self._thr, self._program, name, static=self._static)

    def set_constant(self, name, arr):
        """
        Load a constant array (``arr`` can be either ``numpy`` array or a :py:class:`Array` object)
        corresponding to the symbol ``name`` to device.
        """
        symbol, size = self._program.get_global(name)
        if size != arr.nbytes:
            raise ValueError(
                "Incorrect size of the constant array; "
                "expected " + str(size) + " bytes, got " + str(arr.nbytes))
        if isinstance(arr, self._thr.api.Array):
            self._thr._memcpy_dtod(symbol, arr.base_data, arr.nbytes)
        else:
            # numpy array
            self._thr._memcpy_htod(symbol, arr)


class Kernel:
    """
    An object containing GPU kernel.

    .. py:attribute:: max_work_group_size

        Maximum size of the work group for the kernel.
    """

    def __init__(self, thr, program, name, static=False):
        """__init__()""" # hide the signature from Sphinx

        self._thr = thr
        self._program = program
        self._kernel = self._get_kernel(program, name)
        self._static = static
        self._fill_attributes()

    def prepare(self, global_size, local_size=None, local_mem=0):
        """
        Prepare the kernel for execution with given parameters.

        :param global_size: an integer or a tuple of integers,
            specifying total number of work items to run.
        :param local_size: an integer or a tuple of integers,
            specifying the size of a single work group.
            Should have the same number of dimensions as ``global_size``.
            If ``None`` is passed, some ``local_size`` will be picked internally.
        :param local_mem: (**CUDA API only**) amount of dynamic local memory (in bytes)
        """
        raise NotImplementedError()

    def prepared_call(self, *args):
        """
        Execute the kernel.
        :py:class:`Array` objects are allowed as arguments.
        In case of the OpenCL backend, returns a ``pyopencl.Event`` object.
        """
        result = self._prepared_call(*args)
        self._thr._synchronize()
        return result

    def __call__(self, *args, **kwds):
        """
        A shortcut for successive call to :py:meth:`prepare` and :py:meth:`prepared_call`.
        In case of the OpenCL backend, returns a ``pyopencl.Event`` object.
        """
        if not self._static:
            if 'global_size' in kwds:
                prep_args = (kwds.pop('global_size'),)
            else:
                raise TypeError("global_size keyword argument must be set")
            self.prepare(*prep_args, **kwds)

        return self.prepared_call(*args)

    def set_constant(self, name, arr):
        """
        Load a constant array (``arr`` can be either ``numpy`` array or a :py:class:`Array` object)
        corresponding to the symbol ``name`` to device.
        Note that all the kernels belonging to the same :py:class:`Program` object
        share constant arrays.
        """
        self._program.set_constant(name, arr)


class StaticKernel:
    """
    An object containing a GPU kernel with fixed call sizes.

    .. py:attribute:: source

        Contains the source code of the program.
    """

    def __init__(self, thr, template_src, name, global_size, local_size=None,
            render_args=None, render_kwds=None, fast_math=False, compiler_options=None,
            constant_arrays=None, keep=False):
        """__init__()""" # hide the signature from Sphinx

        self._thr = thr

        if render_args is None:
            render_args = []
        if render_kwds is None:
            render_kwds = {}

        main_src = render_template_source(
            template_src, render_args=render_args, render_kwds=render_kwds)

        # Since virtual size function require some registers, they affect the maximum local size.
        # Start from the device's max work group size as the first approximation
        # and recompile kernels with smaller local sizes until convergence.
        max_local_size = thr.device_params.max_work_group_size

        while True:

            # Try to find kernel launch parameters for the requested local size.
            # May raise OutOfResourcesError if it's not possible,
            # just let it pass to the caller.
            vs = VirtualSizes(
                thr.device_params, global_size,
                virtual_local_size=local_size,
                max_local_size=max_local_size)

            # Try to compile the kernel with the corresponding virtual size functions
            program = Program(
                self._thr, vs.vsize_functions + main_src,
                static=True, fast_math=fast_math, compiler_options=compiler_options,
                constant_arrays=constant_arrays, keep=keep)
            kernel = getattr(program, name)

            if kernel.max_work_group_size >= product(vs.real_local_size):
                # Kernel will execute with this local size, use it
                break

            # By the contract of VirtualSizes,
            # product(vs.real_local_size) <= max_local_size
            # Also, since we're still in this loop,
            # kernel.max_work_group_size < product(vs.real_local_size).
            # Therefore the new max_local_size value is guaranteed
            # to be smaller than the previous one.
            max_local_size = kernel.max_work_group_size

        self._program = program
        self._kernel = kernel
        self.virtual_local_size = vs.virtual_local_size
        self.virtual_global_size = vs.virtual_global_size
        self.local_size = vs.real_local_size
        self.global_size = vs.real_global_size

        self._kernel.prepare(self.global_size, local_size=self.local_size)

    def __call__(self, *args):
        """
        Execute the kernel.
        :py:class:`Array` objects are allowed as arguments.
        In case of the OpenCL backend, returns a ``pyopencl.Event`` object.
        """
        return self._kernel(*args)

    def set_constant(self, name, arr):
        """
        Load a constant array (``arr`` can be either ``numpy`` array or a :py:class:`Array` object)
        corresponding to the symbol ``name`` to device.
        """
        self._program.set_constant(name, arr)
