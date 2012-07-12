import numpy
import pyopencl as cl
import pyopencl.array as clarray
import sys

import tigger.cluda as cluda
import tigger.cluda.dtypes as dtypes


class OclEnvironment:

    def __init__(self, device_num=0, fast_math=True, sync=False):
        self.fast_math = fast_math

        platforms = cl.get_platforms()
        target_device = None
        for p in platforms:
            devices = p.get_devices()
            for d in devices:
                params = OclDeviceParameters(d)
                if params.max_block_size > 1:
                    target_device = d
                    break
            if target_device is not None:
                break

        self.context = cl.Context(devices=[target_device])
        self.queue = cl.CommandQueue(self.context)
        self.sync = sync
        self.params = OclDeviceParameters(self.context.get_info(cl.context_info.DEVICES)[0])

    def supportsDtype(self, dtype):
        if dtypes.is_double(dtype):
            extensions = self.context.devices[0].extensions
            return "cl_khr_fp64" in extensions or "cl_amd_fp64" in extensions
        else:
            return True # TODO: check if there are other limitations

    def allocate(self, shape, dtype):
        return clarray.Array(self.queue, shape, dtype=dtype)

    def fromDevice(self, arr):
        return arr.get()

    def toDevice(self, arr):
        res = clarray.to_device(self.queue, arr)
        if self.sync:
            self.queue.finish()
        return res

    def release(self):
        pass

    def compile(self, src):
        return OclModule(self, src)


class OclDeviceParameters:

    def __init__(self, device):
        self.max_block_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self.max_block_dims = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)

        self.max_grid_dims = [sys.maxint, sys.maxint]

        self.smem_banks = 16 # FIXME: must get it from device
        self.warp_size = 32 # FIXME: must get it from device

        self.api = 'ocl'


class OclModule:

    def __init__(self, env, src):
        self._env = env
        options = "-cl-mad-enable -cl-fast-relaxed-math" if env.fast_math else ""

        try:
            self._module = cl.Program(env.context, src).build(options=options)
        except:
            for i, l in enumerate(src.split('\n')):
                print i, ":", l
            raise

    def __getattr__(self, name):
        return OclKernel(self._env, getattr(self._module, name))


class OclKernel:

    def __init__(self, env, kernel):
        self._env = env
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

        self._kernel(self._env.queue, global_size, block, *args)
