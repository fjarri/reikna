from logging import error
import sys

import numpy
import pyopencl as cl
import pyopencl.array as clarray

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes
from tigger.cluda.kernel import render_prelude, render_template_source


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
                    if params.max_block_size > 1:
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

        self._queue = self.create_stream() if stream is None else stream

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

    def compile_raw(self, src):
        return Module(self, False, src)

    def compile(self, template_src, **kwds):
        return Module(self, True, template_src, **kwds)


class DeviceParameters:

    def __init__(self, device):
        self.max_block_size = device.max_work_group_size
        self.max_block_dims = device.max_work_item_sizes

        self.max_grid_dims = [sys.maxint, sys.maxint]

        # Warp size (nVidia), or wavefront size (AMD), or SIMD width
        # is supposed to be the number of threads that are
        # executed simultaneously on the same computation unit
        # (so you can assume that they are perfectly synchronized).
        # Shared (local for AMD) memory banks is a number of successive 32-bit words
        # you can access without getting bank conflicts.
        if device.type == cl.device_type.CPU:
            # For CPU both values do not make much sense,
            # so we are just setting them to maximum
            self.smem_banks = self.max_block_size
            self.warp_size = self.max_block_size
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
            self.warp_size = 64


class Module:

    def __init__(self, ctx, is_template, src, **kwds):
        self._ctx = ctx
        options = "-cl-mad-enable -cl-fast-relaxed-math" if ctx.fast_math else ""

        prelude = render_prelude(self._ctx)
        if is_template:
            src = render_template_source(src, **kwds)

        # Casting source code to ASCII explicitly
        # New versions of Mako produce Unicode output by default,
        # and it makes OpenCL compiler unhappy
        src = str(prelude + src)

        try:
            self._module = cl.Program(ctx.context, src).build(options=options)
        except:
            listing = "\n".join([str(i+1) + ":" + l for i, l in enumerate(src.split('\n'))])
            error("Failed to compile:\n" + listing)
            raise

    def __getattr__(self, name):
        return Kernel(self._ctx, getattr(self._module, name))


class Kernel:

    def __init__(self, ctx, kernel):
        self._ctx = ctx
        self._kernel = kernel

    def __call__(self, *args, **kwds):
        # Python 2.* cannot handle explicit keywords after variable-length positional arguments
        block = kwds.pop('block', (1, 1, 1))
        grid = kwds.pop('grid', (1, 1))
        shared = kwds.pop('shared', 0)
        assert len(kwds) == 0, "Unknown keyword arguments: " + str(kwds.keys())

        global_size = (block[0] * grid[0], block[1] * grid[1], block[2])

        # Unlike PyCuda, PyOpenCL does not allow passing array objects as is
        args = [x.data if isinstance(x, clarray.Array) else x for x in args]

        self._kernel(self._ctx._queue, global_size, block, *args)
        self._ctx._synchronize()
