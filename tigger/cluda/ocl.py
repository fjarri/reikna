from logging import error
import sys

import numpy
import pyopencl as cl
import pyopencl.array as clarray

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes
from tigger.helpers import wrap_in_tuple, product
from tigger.cluda.kernel import render_prelude, render_template_source
from tigger.cluda.vsize import VirtualSizes, render_stub_vsize_funcs


API_ID = cluda.API_OCL


def get_platforms():
    return cl.get_platforms()


class Context:

    @classmethod
    def create(cls, device=None, **kwds):

        # cl.create_some_context() creates multiple-device context,
        # and we do not want that (yet)

        def find_suitable_device():
            platforms = get_platforms()
            target_device = None
            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    params = DeviceParameters(device)
                    if params.max_work_group_size > 1:
                        return device
            return None

        if device is None:
            device = find_suitable_device()
            if device is None:
                raise RuntimeError("Cannot find suitable OpenCL device to create CLUDA context")

        ctx = cl.Context(devices=[device])
        kwds['owns_context'] = True

        return cls(ctx, **kwds)

    def __init__(self, context, queue=None, fast_math=True, async=True, owns_context=False):
        self.api = cluda.api(API_ID)
        self._fast_math = fast_math
        self._context = context
        self._async = async
        self.device_params = DeviceParameters(context.get_info(cl.context_info.DEVICES)[0])
        self._device = self._context.devices[0]

        self._queue = self.create_queue() if queue is None else queue
        self._released = False if owns_context else True

    def override_device_params(self, **kwds):
        for kwd in kwds:
            if hasattr(self.device_params, kwd):
                setattr(self.device_params, kwd, kwds[kwd])
            else:
                raise ValueError("Device parameter " + str(kwd) + " does not exist")

    def create_queue(self):
        return cl.CommandQueue(self._context)

    def supports_dtype(self, dtype):
        if dtypes.is_double(dtype):
            extensions = self._context.devices[0].extensions
            return "cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions
        else:
            return True

    def allocate(self, size):
        return cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=size)

    def array(self, *args, **kwds):
        return clarray.Array(self._queue, *args, **kwds)

    def empty_like(self, arr):
        return self.array(arr.shape, arr.dtype)

    def to_device(self, arr, dest=None):
        if dest is None:
            arr_device = self.empty_like(arr)
        else:
            arr_device = dest

        arr_device.set(arr, queue=self._queue, async=self._async)

        if dest is None:
            return arr_device

    def from_device(self, arr, dest=None, async=False):
        arr_cpu = arr.get(queue=self._queue, ary=dest, async=async)
        if dest is None:
            return arr_cpu

    def copy_array(self, arr, dest=None, src_offset=0, dest_offset=0, size=None):
        if dest is None:
            arr_device = self.empty_like(arr)
        else:
            arr_device = dest

        itemsize = arr.dtype.itemsize
        nbytes = arr.nbytes if size is None else itemsize * size
        src_offset *= itemsize
        dest_offset *= itemsize

        cl.enqueue_copy(self._queue,
            arr_device.data, arr.data,
            byte_count=nbytes, src_offset=src_offset, dest_offset=dest_offset)
        self._synchronize()

        if dest is None:
            return arr_device

    def synchronize(self):
        self._queue.finish()

    def _synchronize(self):
        if not self._async:
            self.synchronize()

    def release(self):
        if not self._released:
            del self._device
            del self._queue
            del self._context
            self._released = True

    def __del__(self):
        self.release()

    def _compile(self, src):
        options = "-cl-mad-enable -cl-fast-relaxed-math" if self._fast_math else ""
        try:
            module = cl.Program(self._context, src).build(options=options)
        except:
            listing = "\n".join([str(i+1) + ":" + l for i, l in enumerate(src.split('\n'))])
            error("Failed to compile:\n" + listing)
            raise
        return module

    def compile(self, template_src, render_kwds=None):
        return Module(self, template_src, render_kwds=render_kwds)

    def compile_static(self, template_src, name, global_size,
            local_size=None, render_kwds=None):
        return StaticKernel(self, template_src, name, global_size,
            local_size=local_size, render_kwds=render_kwds)


class DeviceParameters:

    def __init__(self, device):

        if device.platform.name == 'Apple' and device.type == cl.device_type.CPU:
        # Apple is being funny again.
        # On OSX 10.8.0 it reports the maximum block size as 1024, when it is really 128.
        # Moreover, if local_barrier() is used in the kernel, it becomes 1
            self.max_work_group_size = 1
            self.max_work_item_sizes = [1, 1, 1]
        else:
            self.max_work_group_size = device.max_work_group_size
            self.max_work_item_sizes = device.max_work_item_sizes

        self.max_num_groups = [sys.maxsize, sys.maxsize, sys.maxsize]

        if device.type == cl.device_type.CPU:
            # For CPU both values do not make much sense
            self.local_mem_banks = self.max_work_group_size
            self.warp_size = 1
        elif "cl_nv_device_attribute_query" in device.extensions:
            # If NV extensions are available, use them to query info
            self.local_mem_banks = 16 if device.compute_capability_major_nv < 2 else 32
            self.warp_size = device.warp_size_nv
        elif device.vendor == 'NVIDIA':
            # nVidia device, but no extensions.
            # Must be APPLE OpenCL implementation.
            self.local_mem_banks = 16
            self.warp_size = 16
        else:
            # AMD card.
            # Do not know how to query this info, so settle for most probable values.

            self.local_mem_banks = 32

            # An alternative is to query CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
            # for some arbitrary kernel.
            self.warp_size = 64

        self.min_mem_coalesce_width = {4: 16, 8: 16, 16: 8}
        self.local_mem_size = device.local_mem_size


class Module:

    def __init__(self, ctx, src, render_kwds=None):
        self._ctx = ctx

        if render_kwds is None:
            render_kwds = {}
        prelude = render_prelude(self._ctx)
        src = render_template_source(src, **render_kwds)

        # Casting source code to ASCII explicitly
        # New versions of Mako produce Unicode output by default,
        # and it makes OpenCL compiler unhappy
        self.source = str(prelude + src)
        self._module = ctx._compile(self.source)

    def __getattr__(self, name):
        return Kernel(self._ctx, getattr(self._module, name))


class Kernel:

    def __init__(self, ctx, kernel):
        self._ctx = ctx
        self._kernel = kernel
        self._max_work_group_size = kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, self._ctx._device)

    def prepare(self, global_size, local_size=None):
        if local_size is None:
            self._local_size = None
        else:
            self._local_size = wrap_in_tuple(local_size)
        self._global_size = wrap_in_tuple(global_size)

    def prepared_call(self, *args):

        # Unlike PyCuda, PyOpenCL does not allow passing array objects as is
        args = [x.data if isinstance(x, clarray.Array) else x for x in args]
        self._kernel(self._ctx._queue, self._global_size, self._local_size, *args)
        self._ctx._synchronize()

    def __call__(self, *args, **kwds):
        if 'global_size' in kwds:
            prep_args = (kwds.pop('global_size'),)
        else:
            prep_args = tuple()
        self.prepare(*prep_args, **kwds)
        self.prepared_call(*args)


class StaticKernel:

    def __init__(self, ctx, src, name, global_size, local_size=None, render_kwds=None):
        self._ctx = ctx

        if render_kwds is None:
            render_kwds = {}

        prelude = render_prelude(self._ctx)
        stub_vsize_funcs = render_stub_vsize_funcs()
        src = render_template_source(src, **render_kwds)

        # We need the first approximation of the maximum thread number for a kernel.
        # Stub virtual size functions instead of real ones will not change it (hopefully).
        stub_module = ctx._compile(str(prelude + stub_vsize_funcs + src))
        stub_kernel = getattr(stub_module, name)
        max_work_group_size = stub_kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, self._ctx._device)

        vs = VirtualSizes(ctx.device_params, max_work_group_size, global_size, local_size)
        static_prelude = vs.render_vsize_funcs()
        self._global_size, self._local_size = vs.get_call_sizes()

        # Casting source code to ASCII explicitly
        # New versions of Mako produce Unicode output by default,
        # and it makes OpenCL compiler unhappy
        self.source = str(prelude + static_prelude + src)
        self._module = ctx._compile(self.source)

        self._kernel = getattr(self._module, name)

        self.max_work_group_size = self._kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, self._ctx._device)
        if self.max_work_group_size < product(self._local_size):
            raise cluda.OutOfResourcesError("Not enough registers/local memory for this local size")

    def __call__(self, *args):
        args = [x.data if isinstance(x, clarray.Array) else x for x in args]
        self._kernel(self._ctx._queue, self._global_size, self._local_size, *args)
        self._ctx._synchronize()
