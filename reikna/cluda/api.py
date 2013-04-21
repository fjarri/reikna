"""
.. py:class:: Buffer

    Low-level untyped memory allocation.
    Actual class depends on the API: :py:class:`pycuda.driver.DeviceAllocation` for ``CUDA``
    and :py:class:`pyopencl.Buffer` for ``OpenCL``.

    .. py:attribute:: size

.. py:class:: Array

    Actual array class is different depending on the API: :py:class:`pycuda.gpuarray.GPUArray`
    for ``CUDA`` and :py:class:`pyopencl.array.Array` for ``OpenCL``.
    This is an interface they both provide.

    .. py:attribute:: shape

    .. py:attribute:: dtype

    .. py:method:: get()

        Returns :py:class:`numpy.ndarray` with the contents of the array.
        Synchronizes the parent :py:class:`~reikna.cluda.api.Thread`.

.. py:class:: DeviceParameters(device)

    An assembly of device parameters necessary for optimizations.

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

from logging import error
import weakref

from reikna.helpers import AttrDict
from reikna.cluda import OutOfResourcesError
from reikna.helpers import wrap_in_tuple, product
from reikna.cluda.kernel import render_prelude, render_template_source
from reikna.cluda.vsize import VirtualSizes
from reikna.cluda.tempalloc import ZeroOffsetManager


def get_id():
    """
    Returns the identifier of this API.
    """
    raise NotImplementedError()


def get_platforms():
    """
    Returns a list of available :py:class:`Platform` objects.
    In case of ``OpenCL`` returned objects are actually instances of :py:class:`pyopencl.Platform`.
    """
    raise NotImplementedError()


class Thread:
    """
    Wraps an existing context in the CLUDA thread object.

    :param cqd: a ``Context``, ``Device`` or ``Stream``/``CommandQueue`` object to base on.
        If a context is passed, a new stream/queue will be created internally.
    :param fast_math: whether to enable fast mathematical operations during compilation.
    :param async: whether to execute all operations with this thread asynchronously
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
    def create(cls, device=None, **kwds):
        """
        Creates a new ``Thread`` object with its own context and queue inside.
        Intended for cases when you want to base your whole program on CLUDA.

        :param device: device to create the thread for, element of the list
            returned by :py:meth:`Platform.get_devices`.
            If not given, the device will be selected internally.
        :type device: :py:class:`pycuda.driver.Device` object for ``CUDA``,
            or :py:class:`pyopencl.Device` object for ``OpenCL``.
        :param kwds: same as in :py:class:`Thread`.
        """

        def find_suitable_device():
            platforms = cls.api.get_platforms()
            target_device = None
            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    params = cls.api.DeviceParameters(device)
                    if params.max_work_group_size > 1:
                        return device
            return None

        if device is None:
            device = find_suitable_device()
            if device is None:
                raise RuntimeError("Cannot find a suitable device to create a CLUDA context")

        return cls.api.Thread(device, **kwds)

    def __init__(self, cqd, fast_math=True, async=True, temp_alloc=None):

        self._async = async
        self._fast_math = fast_math

        self._context, self._queue, self._device, self._owns_context = self._process_cqd(cqd)

        self.device_params = self.api.DeviceParameters(self._device)

        temp_alloc_params = AttrDict(
            cls=ZeroOffsetManager, pack_on_alloc=False, pack_on_free=False)
        if temp_alloc is not None:
            temp_alloc_params.update(temp_alloc)

        self.temp_alloc = temp_alloc_params.cls(weakref.proxy(self),
            pack_on_alloc=temp_alloc_params.pack_on_alloc,
            pack_on_free=temp_alloc_params.pack_on_free)

    def override_device_params(self, **kwds):
        for kwd in kwds:
            if hasattr(self.device_params, kwd):
                setattr(self.device_params, kwd, kwds[kwd])
            else:
                raise ValueError("Device parameter " + str(kwd) + " does not exist")

    def supports_dtype(self, dtype):
        """
        Checks if given ``numpy`` dtype can be used in kernels compiled using this thread.
        """
        raise NotImplementedError()

    def allocate(self, size):
        """
        Creates an untyped memory allocation object of type :py:class:`Buffer` with size ``size``.
        """
        raise NotImplementedError()

    def array(self, shape, dtype, allocator=None):
        """
        Creates an :py:class:`Array` on GPU with given ``shape`` and ``dtype``.
        Optionally, an ``allocator`` is a callable returning any object castable to ``int``
        representing the physical address on the device (for instance, :py:class:`Buffer`).
        """
        raise NotImplementedError()

    def temp_array(self, shape, dtype, dependencies=None):
        """
        Creates an :py:class:`Array` on GPU with given ``shape`` and ``dtype``.
        In order to reduce the memory footprint of the program, the temporary array manager
        will allow these arrays to overlap.
        Two arrays will not overlap, if one of them was specified in ``dependencies``
        for the other one.
        For a list of values ``dependencies`` takes, see the reference entry for
        :py:class:`~reikna.cluda.tempalloc.TemporaryManager`.
        """
        return self.temp_alloc.array(*args, **kwds)

    def empty_like(self, arr):
        """
        Allocates an array on GPU with the same shape and dtype as ``arr``.
        """
        return self.array(arr.shape, arr.dtype)

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

    def from_device(self, arr, dest=None, async=False):
        """
        Transfers the contents of ``arr`` to a :py:class:`numpy.ndarray` object.
        The effect of ``dest`` parameter is the same as in :py:meth:`to_device`.
        If ``async`` is ``True``, the transfer is asynchronous
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

    def _create_program(self, src):
        try:
            program = self._compile(src)
        except:
            listing = "\n".join([str(i+1) + ":" + l for i, l in enumerate(src.split('\n'))])
            error("Failed to compile:\n" + listing)
            raise
        return program

    def compile(self, template_src, render_args=None, render_kwds=None):
        """
        Creates a module object from the given template.

        :param template_src: Mako template source to render
        :param render_kwds: an iterable with positional arguments to pass to the template.
        :param render_kwds: a dictionary with keyword parameters to pass to the template.
        :returns: a :py:class:`Program` object.
        """
        src = render_template_source(
            template_src, render_args=render_args, render_kwds=render_kwds)
        return Program(self, src)

    def compile_static(self, template_src, name, global_size,
            local_size=None, render_args=None, render_kwds=None):
        """
        Creates a kernel object with fixed call sizes,
        which allows to overcome some backend limitations.

        :param template_src: Mako template or a template source to render
        :param name: name of the kernel function
        :param global_size: global size to be used
        :param local_size: local size to be used.
            If ``None``, some suitable one will be picked.
        :param local_mem: (**CUDA API only**) amount of dynamically allocated local memory to be used (in bytes).
        :param render_args: a list of parameters to be passed as positional arguments
            to the template.
        :param render_kwds: a dictionary with additional parameters
            to be used while rendering the template.
        :returns: a :py:class:`StaticKernel` object.
        """
        return StaticKernel(self, template_src, name, global_size,
            local_size=local_size, render_args=render_args, render_kwds=render_kwds)

    def _pytest_finalize_specific(self):
        """
        Overridden by a specific ``Thread`` if it needs to do something before finalizing.
        """
        pass

    def _pytest_finalize(self):
        """
        Py.Test holds the reference to the created funcarg/fixture,
        which interferes with ``__del__`` functionality.
        This method forcefully frees critical resources
        (rendering the object unusable).
        """
        self._pytest_finalize_specific()
        del self._device
        del self._queue
        del self._context


class Program:
    """
    An object with compiled GPU code.

    .. py:attribute:: source

        Contains module source code.

    .. py:attribute:: kernel_name

        Contains :py:class:`Kernel` object for the kernel ``kernel_name``.
    """

    def __init__(self, thr, src, static=False):
        """__init__()""" # hide the signature from Sphinx

        self._thr = thr
        self._static = static

        prelude = render_prelude(self._thr)

        # Casting source code to ASCII explicitly
        # New versions of Mako produce Unicode output by default,
        # and it makes the compiler unhappy
        self.source = str(prelude + src)
        self._program = thr._create_program(self.source)

    def __getattr__(self, name):
        return self._thr.api.Kernel(self._thr, self._program, name, static=self._static)


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
        """
        self._prepared_call(*args)
        self._thr._synchronize()

    def __call__(self, *args, **kwds):
        """
        A shortcut for successive call to :py:meth:`prepare` and :py:meth:`prepared_call`.
        """
        if not self._static:
            if 'global_size' in kwds:
                prep_args = (kwds.pop('global_size'),)
            else:
                raise TypeError("global_size keyword argument must be set")
            self.prepare(*prep_args, **kwds)

        self.prepared_call(*args)


class StaticKernel:
    """
    An object containing a GPU kernel with fixed call sizes.

    .. py:attribute:: source

        Contains the source code of the program.
    """

    def __init__(self, thr, template_src, name, global_size, local_size=None,
            render_args=None, render_kwds=None):
        """__init__()""" # hide the signature from Sphinx

        self._thr = thr

        if render_args is None:
            render_args = []
        if render_kwds is None:
            render_kwds = {}

        main_src = render_template_source(
            template_src, render_args=render_args, render_kwds=render_kwds)

        # We need the first approximation of the maximum thread number for a kernel.
        # Stub virtual size functions instead of real ones will not change it (hopefully).
        stub_vs = VirtualSizes(
            thr.device_params, thr.device_params.max_work_group_size,
            global_size, local_size)
        stub_vsize_funcs = stub_vs.render_vsize_funcs()

        stub_program = Program(self._thr, stub_vsize_funcs + main_src, static=True)
        stub_kernel = getattr(stub_program, name)
        max_work_group_size = stub_kernel.max_work_group_size

        # Second pass, compiling the actual kernel

        vs = VirtualSizes(thr.device_params, max_work_group_size, global_size, local_size)
        vsize_funcs = vs.render_vsize_funcs()
        gs, ls = vs.get_call_sizes()
        self._program = Program(self._thr, vsize_funcs + main_src, static=True)
        self._kernel = getattr(self._program, name)

        if self._kernel.max_work_group_size < product(ls):
            raise cluda.OutOfResourcesError("Not enough registers/local memory for this local size")

        self._kernel.prepare(gs, local_size=ls)

    def __call__(self, *args):
        """
        Execute the kernel.
        :py:class:`Array` objects are allowed as arguments.
        """
        self._kernel(*args)
