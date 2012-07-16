import numpy
import sys
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes


class CudaEnvironment:

    def __init__(self, device_num=0, fast_math=True, sync=False):
        self.fast_math = fast_math
        cuda.init()
        self.context = make_default_context()
        self.params = CudaDeviceParameters(self.context.get_device())
        self.stream = None if sync else cuda.Stream()
        self.sync = sync

        self._released = False

    def supportsDtype(self, dtype):
        if dtypes.is_double(dtype):
            major, minor = self.context.get_device().compute_capability()
            return (major == 1 and minor == 3) or major >= 2
        else:
            return True

    def allocate(self, shape, dtype):
        return gpuarray.GPUArray(shape, dtype=dtype)

    def synchronize(self):
        if not self.sync:
            self.stream.synchronize()

    def fromDevice(self, arr):
        return arr.get()

    def toDevice(self, arr):
        if self.sync:
            return gpuarray.to_gpu(arr)
        else:
            return gpuarray.to_gpu_async(arr, stream=self.stream)

    def compile(self, src):
        return CudaModule(self, src)

    def release(self):
        if not self._released:
            self.context.pop()
            self._released = True

    def __del__(self):
        self.release()


class CudaDeviceParameters:

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

        self.api = 'cuda'


class CudaModule:

    def __init__(self, env, src):
        self._env = env
        options = ['-use_fast_math'] if self._env.fast_math else []

        try:
            self._module = SourceModule(src, no_extern_c=True, options=options)
        except:
            for i, l in enumerate(src.split('\n')):
                print i + 1, ":", l
            raise

    def __getattr__(self, name):
        return CudaKernel(self._env, self._module.get_function(name))


class CudaKernel:

    def __init__(self, env, kernel):
        self._env = env
        self._kernel = kernel

    def __call__(self, *args, **kwds):
        # Python 2.* cannot handle explicit keywords after variable-length positional arguments
        block = kwds.pop('block', (1, 1, 1))
        grid = kwds.pop('grid', (1, 1))
        shared = kwds.pop('shared', 0)
        assert len(kwds) == 0, "Unknown keyword arguments: " + str(kwds.keys())

        self._kernel(*args, grid=grid, block=block, stream=self._env.stream)

