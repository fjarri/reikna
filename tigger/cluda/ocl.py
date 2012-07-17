from logging import error
import sys

import numpy
import pyopencl as cl
import pyopencl.array as clarray

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes


API_ID = cluda.API_OCL


class Context:

    @classmethod
    def create(cls, **kwds):

        # cl.create_some_context() creates multiple-device context,
        # and we do not want that (yet)

        platforms = cl.get_platforms()
        target_device = None
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                params = DeviceParameters(device)
                if params.max_block_size > 1:
                    target_device = device
                    break
            if target_device is not None:
                break

        ctx = cl.Context(devices=[target_device])
        queue = cl.CommandQueue(ctx)

        return cls(ctx, queue, **kwds)

    def __init__(self, context, queue, fast_math=True, sync=False):
        self.api = cluda.api(API_ID)
        self.fast_math = fast_math
        self.context = context
        self.queue = queue
        self.sync = sync
        self.device_params = DeviceParameters(context.get_info(cl.context_info.DEVICES)[0])

    def supports_dtype(self, dtype):
        if dtypes.is_double(dtype):
            extensions = self.context.devices[0].extensions
            return "cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions
        else:
            return True

    def allocate(self, shape, dtype):
        return clarray.Array(self.queue, shape, dtype=dtype)

    def from_device(self, arr):
        return arr.get()

    def synchronize(self):
        self.queue.finish()

    def _synchronize(self):
        if not self.sync:
            self.synchronize()

    def to_device(self, arr):
        res = clarray.to_device(self.queue, arr)
        self._synchronize()
        return res

    def release(self):
        pass

    def compile(self, src):
        return Module(self, src)


class DeviceParameters:

    def __init__(self, device):
        self.max_block_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self.max_block_dims = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)

        self.max_grid_dims = [sys.maxint, sys.maxint]

        self.smem_banks = 16
        self.warp_size = 32


class Module:

    def __init__(self, ctx, src):
        self._ctx = ctx
        options = "-cl-mad-enable -cl-fast-relaxed-math" if ctx.fast_math else ""

        # Casting source code to ASCII explicitly
        # New versions of Mako produce Unicode output by default,
        # and it makes OpenCL compiler unhappy
        src = str(src)

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

        self._kernel(self._ctx.queue, global_size, block, *args)
        self._ctx._synchronize()
