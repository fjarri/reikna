from logging import error

import numpy
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes
from tigger.cluda.kernel import render_prelude, render_template_source


cuda.init()

API_ID = cluda.API_CUDA


class Context:

    @classmethod
    def create(cls, **kwds):
        ctx = make_default_context()
        stream = cuda.Stream()
        kwds['owns_context'] = True
        return cls(ctx, stream, **kwds)

    def __init__(self, context, stream=None, fast_math=True, async=True, owns_context=False):
        self.api = cluda.api(API_ID)
        self.fast_math = fast_math
        self.context = context
        self.async = async
        self.device_params = DeviceParameters(context.get_device())

        self._stream = self.create_stream() if stream is None else stream
        self._released = False if owns_context else True

    def create_stream(self):
        return cuda.Stream()

    def supports_dtype(self, dtype):
        if dtypes.is_double(dtype):
            major, minor = self.context.get_device().compute_capability()
            return (major == 1 and minor == 3) or major >= 2
        else:
            return True

    def allocate(self, shape, dtype):
        return gpuarray.GPUArray(shape, dtype=dtype)

    def empty_like(self, arr):
        return self.allocate(arr.shape, arr.dtype)

    def from_device(self, arr):
        return arr.get()

    def synchronize(self):
        self._stream.synchronize()

    def _synchronize(self):
        if not self.async:
            self.synchronize()

    def to_device(self, arr):
        if self.async:
            return gpuarray.to_gpu_async(arr, stream=self._stream)
        else:
            return gpuarray.to_gpu(arr)

    def compile_raw(self, src):
        return Module(self, False, src)

    def compile(self, template_src, **kwds):
        return Module(self, True, template_src, **kwds)

    def release(self):
        if not self._released:
            self.context.detach()
            self._released = True

    def __del__(self):
        self.release()


class DeviceParameters:

    def __init__(self, device):

        self.max_block_size = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        self.max_block_dims = [
            device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_X),
            device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Y),
            device.get_attribute(cuda.device_attribute.MAX_BLOCK_DIM_Z)
        ]

        self.max_grid_dims = [
            device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_X),
            device.get_attribute(cuda.device_attribute.MAX_GRID_DIM_Y)
        ]

        self.smem_banks = 16
        self.warp_size = device.get_attribute(cuda.device_attribute.WARP_SIZE)


class Module:

    def __init__(self, ctx, is_template, src, **kwds):
        self._ctx = ctx
        options = ['-use_fast_math'] if self._ctx.fast_math else []

        prelude = render_prelude(self._ctx)

        if is_template:
            src = render_template_source(src, **kwds)

        try:
            self._module = SourceModule(prelude + src, no_extern_c=True, options=options)
        except:
            listing = "\n".join([str(i+1) + ":" + l for i, l in enumerate(src.split('\n'))])
            error("Failed to compile:\n" + listing)
            raise

    def __getattr__(self, name):
        return Kernel(self._ctx, self._module.get_function(name))


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

        self._kernel(*args, grid=grid, block=block, stream=self._ctx._stream, shared=shared)

