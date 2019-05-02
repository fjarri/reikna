import sys
from tempfile import mkdtemp
import os.path

import pyopencl as cl
import pyopencl.array as clarray

from reikna.helpers import wrap_in_tuple, product, min_buffer_size
import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes
import reikna.cluda.api as api_base

from reikna.cluda.array_helpers import setitem_method, get_method, roll_method


def get_id():
    return cluda.ocl_id()


def get_platforms():
    return cl.get_platforms()


class Array(clarray.Array):
    """
    A subclass of PyOpenCL ``Array``, with some additional functionality.
    """
    def __init__(
            self, thr, shape, dtype, strides=None, offset=0, nbytes=None,
            allocator=None, base_data=None):
        clarray.Array.__init__(
            self, thr._queue, shape, dtype, strides=strides, allocator=allocator,
            data=base_data, offset=offset)
        self.nbytes = nbytes
        self.thread = thr

    def _new_like_me(self, dtype=None, queue=None):
        """
        Called by PyOpenCL to store the results of arithmetic operations
        or when the array is copied, to make an empty array.
        Need to intercept it to preserve the array type.
        The `queue` argument is ignored, we're always using the queue of the thread.
        """
        return (self.thread.empty_like(self)
                if dtype is None
                else self.thread.array(self.shape, dtype))

    def __getitem__(self, index):
        res = clarray.Array.__getitem__(self, index)

        # Let cl.Array calculate the new strides and offset
        return self.thread.array(
            shape=res.shape, dtype=res.dtype, strides=res.strides,
            base_data=res.base_data,
            offset=res.offset)

    def __setitem__(self, index, value):
        setitem_method(self, index, value)

    def roll(self, shift, axis=-1):
        roll_method(self, shift, axis=axis)

    def get(self):
        if self.flags.forc:
            return clarray.Array.get(self)
        else:
            return get_method(self)

    def _tempalloc_update_buffer(self, data):
        self.base_data = data


class Thread(api_base.Thread):

    api = sys.modules[__name__]

    def _process_cqd(self, cqd):
        if isinstance(cqd, cl.Device):
            context = cl.Context(devices=[cqd])
            return context, cl.CommandQueue(context), cqd, False
        elif isinstance(cqd, cl.Context):
            return cqd, cl.CommandQueue(cqd), cqd.devices[0], False
        elif isinstance(cqd, cl.CommandQueue):
            return cqd.context, cqd, cqd.device, False
        else:
            return ValueError("The value provided is not Device, Context or CommandQueue")

    def array(
            self, shape, dtype, strides=None, offset=0, nbytes=None,
            allocator=None, base=None, base_data=None):

        if allocator is None:
            allocator = self.allocate

        dtype = dtypes.normalize_type(dtype)
        shape = wrap_in_tuple(shape)
        if nbytes is None:
            nbytes = int(min_buffer_size(shape, dtype.itemsize, strides=strides, offset=offset))

        if (offset != 0 or strides is not None) and base_data is None and base is None:
            base_data = allocator(nbytes)
        elif base is not None:
            base_data = base.data

        return Array(
            self, shape, dtype, strides=strides, offset=offset,
            allocator=allocator, base_data=base_data, nbytes=nbytes)

    def allocate(self, size):
        return cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=size)

    def _copy_array(self, dest, src):
        dest.set(src, queue=self._queue, async_=self._async)

    def from_device(self, arr, dest=None, async_=False):
        arr_cpu = arr.get(queue=self._queue, ary=dest, async_=async_)
        if dest is None:
            return arr_cpu

    def _copy_array_buffer(self, dest, src, nbytes, src_offset=0, dest_offset=0):
        cl.enqueue_copy(
            self._queue, dest.data, src.data,
            byte_count=nbytes, src_offset=src_offset, dest_offset=dest_offset)

    def synchronize(self):
        self._queue.finish()

    def _compile(self, src, fast_math=False, compiler_options=None, keep=False):
        options = "-cl-mad-enable -cl-fast-relaxed-math" if fast_math else ""
        if compiler_options is not None:
            options += " " + " ".join(compiler_options)

        if keep:
            temp_dir = mkdtemp()
            temp_file_path = os.path.join(temp_dir, 'kernel.cl')

            with open(temp_file_path, 'w') as f:
                f.write(src)

            print("*** compiler output in", temp_dir)

        else:
            temp_dir = None

        return cl.Program(self._context, src).build(options=options, cache_dir=temp_dir)


class DeviceParameters:

    def __init__(self, device):

        self.api_id = get_id()

        self._device = device

        if device.platform.name == 'Apple' and device.type == cl.device_type.CPU:
        # Apple is being funny again.
        # On OSX 10.8.0 it reports the maximum block size as 1024, when it is really 128.
        # Moreover, if local_barrier() is used in the kernel, it becomes 1
            self.max_work_group_size = 1
            self.max_work_item_sizes = [1, 1, 1]
        else:
            self.max_work_group_size = device.max_work_group_size
            self.max_work_item_sizes = device.max_work_item_sizes

        max_size = 2 ** device.address_bits
        self.max_num_groups = [max_size, max_size, max_size]

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

        self.compute_units = device.max_compute_units

    def supports_dtype(self, dtype):
        if dtypes.is_double(dtype):
            extensions = self._device.extensions
            return "cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions
        else:
            return True


class Kernel(api_base.Kernel):

    def _get_kernel(self, program, name):
        return getattr(program, name)

    def _fill_attributes(self):
        self.max_work_group_size = self._kernel.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, self._thr._device)

    def prepare(self, global_size, local_size=None, local_mem=0):
        # ``local_mem`` is ignored, since it cannot be easily passed to the kernel
        # (a special kernel argument is requred).
        if local_size is None:
            self._local_size = None
        else:
            self._local_size = wrap_in_tuple(local_size)
        self._global_size = wrap_in_tuple(global_size)

    def _prepared_call(self, *args):
        # Passing base_data, assuming that the kernel knows how to handle the offset and the strides
        args = [x.base_data if isinstance(x, clarray.Array) else x for x in args]
        return self._kernel(self._thr._queue, self._global_size, self._local_size, *args)
