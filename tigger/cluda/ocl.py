from logging import error
import sys

import numpy
import pyopencl as cl
import pyopencl.array as clarray

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes
from tigger.cluda.helpers import wrap_in_tuple
from tigger.cluda.kernel import render_prelude, render_template_source
from tigger.cluda.vsize import VirtualSizes


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

        return cls(ctx, **kwds)

    def __init__(self, context, stream=None, fast_math=True, async=True):
        self.api = cluda.api(API_ID)
        self.fast_math = fast_math
        self.context = context
        self.async = async
        self.device_params = DeviceParameters(context.get_info(cl.context_info.DEVICES)[0])
        self.device = self.context.devices[0]

        self._queue = self.create_stream() if stream is None else stream

    def override_device_params(self, **kwds):
        for kwd in kwds:
            if hasattr(self.device_params, kwd):
                setattr(self.device_params, kwd, kwds[kwd])
            else:
                raise ValueError("Device parameter " + str(kwd) + " does not exist")

    def create_stream(self):
        return cl.CommandQueue(self.context)

    def supports_dtype(self, dtype):
        if dtypes.is_double(dtype):
            extensions = self.context.devices[0].extensions
            return "cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions
        else:
            return True

    def allocate(self, shape, dtype):
        return clarray.Array(self._queue, shape, dtype=dtype)

    def empty_like(self, arr):
        return self.allocate(arr.shape, arr.dtype)

    def to_device(self, arr, dest=None):
        if dest is None:
            arr_device = self.empty_like(arr)
        else:
            arr_device = dest

        arr_device.set(arr, queue=self._queue, async=self.async)

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
        if not self.async:
            self.synchronize()

    def release(self):
        pass

    def _compile(self, src):
        options = "-cl-mad-enable -cl-fast-relaxed-math" if self.fast_math else ""
        try:
            module = cl.Program(self.context, src).build(options=options)
        except:
            listing = "\n".join([str(i+1) + ":" + l for i, l in enumerate(src.split('\n'))])
            error("Failed to compile:\n" + listing)
            raise
        return module

    def compile(self, template_src, render_kwds=None):
        return Module(self, template_src, render_kwds=render_kwds)

    def compile_static(self, template_src, name, global_size,
            local_size=None, shared=0, render_kwds=None):
        return StaticKernel(self, template_src, name, global_size,
            local_size=local_size, shared=shared, render_kwds=render_kwds)


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

        self.max_grid_sizes = [sys.maxint, sys.maxint, sys.maxint]

        if device.type == cl.device_type.CPU:
            # For CPU both values do not make much sense,
            # so we are just setting them to maximum
            self.smem_banks = self.max_work_group_size
            self.warp_size = self.max_work_group_size
        elif "cl_nv_device_attribute_query" in device.extensions:
            # If NV extensions are available, use them to query info
            self.smem_banks = 16 if device.compute_capability_major_nv < 2 else 32
            self.warp_size = device.warp_size_nv
        elif device.vendor == 'NVIDIA':
            # nVidia device, but no extensions.
            # Must be APPLE OpenCL implementation.
            self.smem_banks = 16
            self.warp_size = 16
        else:
            # AMD card.
            # Do not know how to query this info, so settle for most probable values.

            self.smem_banks = 32

            # An alternative is to query CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
            # for some arbitrary kernel.
            self.warp_size = 64


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
        self.max_work_group_size = kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, self._ctx.device)

    def prepare(self, global_size, local_size=None, shared=0):
        if local_size is None:
            self.local_size = None
        else:
            self.local_size = wrap_in_tuple(local_size)
        self.global_size = wrap_in_tuple(global_size)
        self.shared = shared

    def prepared_call(self, *args):

        # Unlike PyCuda, PyOpenCL does not allow passing array objects as is
        args = [x.data if isinstance(x, clarray.Array) else x for x in args]
        self._kernel(self._ctx._queue, self.global_size, self.local_size, *args)
        self._ctx._synchronize()

    def __call__(self, *args, **kwds):
        if 'global_size' in kwds:
            prep_args = (kwds.pop('global_size'),)
        else:
            prep_args = tuple()
        self.prepare(*prep_args, **kwds)
        self.prepared_call(*args)


class StaticKernel:

    def __init__(self, ctx, src, name, global_size, local_size=None, shared=0, render_kwds=None):
        self._ctx = ctx

        if render_kwds is None:
            render_kwds = {}

        prelude = render_prelude(self._ctx)

        vs = VirtualSizes(ctx.device_params, global_size, local_size)
        static_prelude = vs.render_vsize_funcs()
        self.global_size, self.local_size = vs.get_call_sizes()
        self.shared = shared

        src = render_template_source(src, **render_kwds)

        # Casting source code to ASCII explicitly
        # New versions of Mako produce Unicode output by default,
        # and it makes OpenCL compiler unhappy
        self.source = str(prelude + static_prelude + src)
        self._module = ctx._compile(self.source)

        self._kernel = getattr(self._module, name)

    def __call__(self, *args):
        args = [x.data if isinstance(x, clarray.Array) else x for x in args]
        self._kernel(self._ctx._queue, self.global_size, self.local_size, *args)
        self._ctx._synchronize()
