import sys

import pyopencl as cl
import pyopencl.array as clarray

from reikna.helpers import wrap_in_tuple
import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes
import reikna.cluda.api as api_base


def get_id():
    return cluda.ocl_id()


def get_platforms():
    return cl.get_platforms()


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

    def allocate(self, size):
        return cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=size)

    def array(self, shape, dtype, strides=None, allocator=None):
        return clarray.Array(self._queue, shape, dtype, strides=strides, allocator=allocator)

    def _copy_array(self, dest, src):
        dest.set(src, queue=self._queue, async=self._async)

    def from_device(self, arr, dest=None, async=False):
        arr_cpu = arr.get(queue=self._queue, ary=dest, async=async)
        if dest is None:
            return arr_cpu

    def _copy_array_buffer(self, dest, src, nbytes, src_offset=0, dest_offset=0):
        cl.enqueue_copy(
            self._queue, dest.data, src.data,
            byte_count=nbytes, src_offset=src_offset, dest_offset=dest_offset)

    def synchronize(self):
        self._queue.finish()

    def _compile(self, src, fast_math=False):
        options = "-cl-mad-enable -cl-fast-relaxed-math" if fast_math else ""
        return cl.Program(self._context, src).build(options=options)


class DeviceParameters:

    def __init__(self, device):

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
        args = [x.data if isinstance(x, clarray.Array) else x for x in args]
        self._kernel(self._thr._queue, self._global_size, self._local_size, *args)
