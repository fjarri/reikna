from logging import error

import numpy
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes
from tigger.cluda.kernel import render_prelude, render_template_source


cuda.init()

API_ID = cluda.API_CUDA


def get_platforms():
    # For CUDA, there's only one platform
    return [Platform()]


class Platform:
    """
    Mimics pyopencl.Platform
    """

    name = "nVidia CUDA"
    vendor = "nVidia"
    version = ".".join(str(x) for x in cuda.get_version())

    def get_devices(self):
        return [cuda.Device(num) for num in xrange(cuda.Device.count())]

    def __str__(self):
        return self.name + " " + self.version


class Context:

    @classmethod
    def create(cls, device=None, **kwds):

        if device is None:
            platform = get_platforms()[0]
            device = platform.get_devices()[0]

        ctx = device.make_context()
        kwds['owns_context'] = True
        return cls(ctx, **kwds)

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

    def to_device(self, arr, dest=None):
        if dest is None:
            arr_device = self.empty_like(arr)
        else:
            arr_device = dest

        arr_device.set_async(arr, stream=self._stream)
        self._synchronize()

        if dest is None:
            return arr_device

    def from_device(self, arr, dest=None, async=False):
        if async:
            arr_cpu = arr.get_async(ary=dest, stream=self._stream)
        else:
            arr_cpu = arr.get(ary=dest)

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

        cuda.memcpy_dtod_async(int(arr_device.gpudata) + dest_offset,
            int(arr.gpudata) + src_offset,
            nbytes, stream=self._stream)
        self._synchronize()

        if dest is None:
            return arr_device

    def synchronize(self):
        self._stream.synchronize()

    def _synchronize(self):
        if not self.async:
            self.synchronize()

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

        self.max_work_group_size = device.max_threads_per_block
        self.max_work_item_sizes = [
            device.max_block_dim_x,
            device.max_block_dim_y,
            device.max_block_dim_z]

        self.max_grid_dims = [
            device.max_grid_dim_x,
            device.max_grid_dim_y]

        # there is no corresponding constant in the API at the moment
        self.smem_banks = 16 if device.compute_capability()[0] < 2 else 32

        self.warp_size = device.warp_size


class Module:

    def __init__(self, ctx, is_template, src, **kwds):
        self._ctx = ctx
        options = ['-use_fast_math'] if self._ctx.fast_math else []

        prelude = render_prelude(self._ctx)

        if is_template:
            src = render_template_source(src, **kwds)
        src = prelude + src

        try:
            self._module = SourceModule(src, no_extern_c=True, options=options)
        except:
            listing = "\n".join([str(i+1) + ":" + l for i, l in enumerate(src.split('\n'))])
            error("Failed to compile:\n" + listing)
            raise

    def __getattr__(self, name):
        return Kernel(self._ctx, self._module.get_function(name))


def bounding_grid(N, Ms):
    """
    For a natural N and M_1, M_2, ... returns (n_1, n_2, ...) such that:
    1) n_1 * n_2 * ... >= N
    2) n_i <= M_i
    3) n_1 * n_2 * ... = min
    """

    product = lambda l: reduce(lambda x, y: x * y, l, 1)
    assert product(Ms) >= N

    # stupid algorithm, just for stub
    if N < Ms[0]:
        return [N]

    dims = len(Ms)
    n = int(N ** (1. / dims)) + 1
    return [n] * dims


class Kernel:

    def __init__(self, ctx, kernel):
        self._ctx = ctx
        self._kernel = kernel

    def prepare(self, global_size, local_size=None, shared=0):
        self.shared = shared

        if isinstance(global_size, int):
            # Flat indices mode.
            # In order to emulate it in CUDA, we need the user
            # to skip threads with idx > size manually.

            if local_size is not None:
                if not isinstance(local_size, int):
                    if len(local_size) > 1:
                        raise ValueError("Global/local work sizes have differing dimensions")
                    local_size = local_size[0]
            else:
                # temporary stub
                local_size = min(self._ctx.device_params.max_work_group_size, global_size)

            # since there is a maximum on a grid width, we need to pick a pair gx, gy
            # so that gx * gy >= grid and gx * gy is minimal.
            grid_size = (global_size - 1) / local_size + 1
            self.grid = tuple(bounding_grid(grid_size, self._ctx.device_params.max_grid_dims))
            self.block = (local_size, 1, 1)
        else:
            if local_size is None:
                raise NotImplementedError(
                    "Automatic local size with non-flat global size is not supported")

            if isinstance(local_size, int):
                local_size = (local_size,)

            if len(local_size) != len(global_size):
                raise ValueError("Global/local work sizes have differing dimensions")

            grid = []
            for gs, ls in zip(global_size, local_size):
                if gs % ls != 0:
                    raise ValueError("Global sizes must be multiples of corresponding local sizes")
                grid.append(gs / ls)

            self.block = local_size
            self.grid = tuple(grid)

    def prepared_call(self, *args):
        self._kernel(*args, grid=self.grid, block=self.block,
            stream=self._ctx._stream, shared=self.shared)
        self._ctx._synchronize()

    def __call__(self, *args, **kwds):
        # Python 2.* cannot handle explicit keywords after variable-length positional arguments
        self.prepare(**kwds)
        self.prepared_call(*args)
