import sys
import itertools

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData

import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes
from reikna.helpers import factors, wrap_in_tuple, product
import reikna.cluda.api as api_base


cuda.init()


def get_id():
    return cluda.cuda_id()


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
        return [Device(num) for num in range(cuda.Device.count())]

    def __str__(self):
        return self.name + " " + self.version


class Device(cuda.Device):

    def __init__(self, device_num):
        cuda.Device.__init__(self, device_num)
        self.name = self.name()


class Buffer:
    """
    Mimics pyopencl.Buffer
    """

    def __init__(self, size):
        self._buffer = cuda.mem_alloc(size)
        self.size = size

    def __int__(self):
        return int(self._buffer)

    def __long__(self):
        return long(self._buffer)

    def __del__(self):
        self._buffer.free()


class Thread(api_base.Thread):

    api = sys.modules[__name__]

    def __init__(self, *args, **kwds):
        api_base.Thread.__init__(self, *args, **kwds)
        self._active = True

    def _process_cqd(self, cqd):
        if isinstance(cqd, cuda.Device):
            context = cqd.make_context()
            stream = cuda.Stream()
            return context, stream, cqd, True
        elif isinstance(cqd, cuda.Context) or cqd is None:
            return cqd, cuda.Stream(), cqd.get_device(), False
        elif isinstance(cqd, cuda.Stream):
            # There's no function in PyCUDA to get the current context,
            # but we do not really need it anyway.
            return None, cqd, cuda.Context.get_device(), False
        else:
            return ValueError("The value provided is not Device, Context or Stream")

    def allocate(self, size):
        return Buffer(size)

    def array(self, shape, dtype, strides=None, allocator=None):
        # In PyCUDA, the default allocator is not None, but a default alloc object
        kwds = {}
        if strides is not None:
            kwds['strides'] = strides
        if allocator is not None:
            kwds['allocator'] = allocator
        return gpuarray.GPUArray(shape, dtype, **kwds)

    def _copy_array(self, dest, src):
        dest.set_async(src, stream=self._queue)

    def from_device(self, arr, dest=None, async=False):
        if async:
            arr_cpu = arr.get_async(ary=dest, stream=self._queue)
        else:
            arr_cpu = arr.get(ary=dest)

        if dest is None:
            return arr_cpu

    def _copy_array_buffer(self, dest, src, nbytes, src_offset=0, dest_offset=0):
        cuda.memcpy_dtod_async(
            int(dest.gpudata) + dest_offset,
            int(src.gpudata) + src_offset,
            nbytes, stream=self._queue)

    def synchronize(self):
        self._queue.synchronize()

    def _compile(self, src, fast_math=False):
        options = ['-use_fast_math'] if fast_math else []
        return SourceModule(src, no_extern_c=True, options=options)

    def _cuda_push(self):
        assert not self._active
        self._context.push()
        self._active = True

    def _cuda_pop(self):
        assert self._active
        cuda.Context.pop()
        self._active = False

    def _release_specific(self):
        # If we own the context, it is our responsibility to pop() it
        if self._owns_context and self._active:
            cuda.Context.pop()
        if self._owns_context:
            self._context.detach()

    def __del__(self):
        self.release()


class DeviceParameters:

    def __init__(self, device):

        self._device = device
        self.max_work_group_size = device.max_threads_per_block
        self.max_work_item_sizes = [
            device.max_block_dim_x,
            device.max_block_dim_y,
            device.max_block_dim_z]

        self.max_num_groups = [
            device.max_grid_dim_x,
            device.max_grid_dim_y,
            device.max_grid_dim_z]

        # there is no corresponding constant in the API at the moment
        self.local_mem_banks = 16 if device.compute_capability()[0] < 2 else 32

        self.warp_size = device.warp_size

        devdata = DeviceData(device)
        self.min_mem_coalesce_width = {
            size:devdata.align_words(word_size=size) for size in [4, 8, 16]}
        self.local_mem_size = device.max_shared_memory_per_block

    def supports_dtype(self, dtype):
        if dtypes.is_double(dtype):
            major, minor = self._device.compute_capability()
            return (major == 1 and minor == 3) or major >= 2
        else:
            return True


def find_local_size(global_size, max_work_item_sizes, max_work_group_size):
    """
    Mimics the OpenCL local size finding algorithm.
    Returns the tuple of the same length as ``global_size``, with every element
    being a factor of the corresponding element of ``global_size``.
    Neither of the elements of ``local_size`` are greater then the corresponding element
    of ``max_work_item_sizes``, and their product is not greater than ``max_work_group_size``.
    """
    if len(global_size) == 0:
        return tuple()

    if max_work_group_size == 1:
        return (1,) * len(global_size)

    gs_factors = factors(global_size[0], limit=min(max_work_item_sizes[0], max_work_group_size))
    local_size_1d, _ = gs_factors[-1]
    remainder = find_local_size(
        global_size[1:], max_work_item_sizes[1:], max_work_group_size // local_size_1d)

    return (local_size_1d,) + remainder


class Kernel(api_base.Kernel):

    def _get_kernel(self, program, name):
        return program.get_function(name)

    def _fill_attributes(self):
        self.max_work_group_size = self._kernel.get_attribute(
            cuda.function_attribute.MAX_THREADS_PER_BLOCK)

    def prepare(self, global_size, local_size=None, local_mem=0):
        global_size = wrap_in_tuple(global_size)
        self._local_mem = local_mem

        max_dims = self._thr.device_params.max_work_item_sizes
        if len(global_size) > len(max_dims):
            raise ValueError("Global size has too many dimensions")

        if local_size is not None:
            local_size = wrap_in_tuple(local_size)
            if len(local_size) != len(global_size):
                raise ValueError("Global/local work sizes have differing dimensions")
        else:
            local_size = find_local_size(global_size, max_dims, self.max_work_group_size)

        grid = []
        for gs, ls in zip(global_size, local_size):
            if gs % ls != 0:
                raise ValueError("Global sizes must be multiples of corresponding local sizes")
            grid.append(gs // ls)

        # append missing dimensions, otherwise PyCUDA will complain
        self._local_size = local_size + (1,) * (3 - len(grid))
        self._grid = tuple(grid) + (1,) * (3 - len(grid))

    def prepared_call(self, *args):
        self._kernel(*args, grid=self._grid, block=self._local_size,
            stream=self._thr._queue, shared=self._local_mem)
