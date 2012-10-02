import numpy

from tigger.helpers import *
from tigger.core import *
import tigger.cluda.dtypes as dtypes
from tigger.cluda import OutOfResourcesError

TEMPLATE = template_for(__file__)


MAX_RADIX = 16


def get_radix_array(n, use_max_radix=False):
    """
    For any n, this function decomposes n into factors for loacal memory tranpose
    based fft. Factors (radices) are sorted such that the first one (radix_array[0])
    is the largest. This base radix determines the number of registers used by each
    work item and product of remaining radices determine the size of work group needed.
    To make things concrete with and example, suppose n = 1024. It is decomposed into
    1024 = 16 x 16 x 4. Hence kernel uses float2 a[16], for local in-register fft and
    needs 16 x 4 = 64 work items per work group. So kernel first performance 64 length
    16 ffts (64 work items working in parallel) following by transpose using local
    memory followed by again 64 length 16 ffts followed by transpose using local memory
    followed by 256 length 4 ffts. For the last step since with size of work group is
    64 and each work item can array for 16 values, 64 work items can compute 256 length
    4 ffts by each work item computing 4 length 4 ffts.
    Similarly for n = 2048 = 8 x 8 x 8 x 4, each work group has 8 x 8 x 4 = 256 work
    iterms which each computes 256 (in-parallel) length 8 ffts in-register, followed
    by transpose using local memory, followed by 256 length 8 in-register ffts, followed
    by transpose using local memory, followed by 256 length 8 in-register ffts, followed
    by transpose using local memory, followed by 512 length 4 in-register ffts. Again,
    for the last step, each work item computes two length 4 in-register ffts and thus
    256 work items are needed to compute all 512 ffts.
    For n = 32 = 8 x 4, 4 work items first compute 4 in-register
    lenth 8 ffts, followed by transpose using local memory followed by 8 in-register
    length 4 ffts, where each work item computes two length 4 ffts thus 4 work items
    can compute 8 length 4 ffts. However if work group size of say 64 is choosen,
    each work group can compute 64/ 4 = 16 size 32 ffts (batched transform).
    Users can play with these parameters to figure what gives best performance on
    their particular device i.e. some device have less register space thus using
    smaller base radix can avoid spilling ... some has small local memory thus
    using smaller work group size may be required etc
    """
    if use_max_radix:
        radix = min(n, MAX_RADIX)
        radix_array = []
        while n > radix:
            radix_array.append(radix)
            n /= radix
        radix_array.append(n)
        return radix_array
    else:
        arrays = {
            2: [2], 4: [4], 8: [8],
            16: [8, 2], 32: [8, 4], 64: [8, 8],
            128: [8, 4, 4],
            256: [4, 4, 4, 4],
            512: [8, 8, 8],
            1024: [16, 16, 4],
            2048: [8, 8, 8, 4]
        }
        if n in arrays:
            return arrays[n]
        else:
            raise ValueError("Wrong problem size: " + str(n))


def get_global_radix_info(n):
    """
    For n larger than what can be computed using local memory fft, global transposes
    multiple kernel launces is needed. For these sizes, n can be decomposed using
    much larger base radices i.e. say n = 262144 = 128 x 64 x 32. Thus three kernel
    launches will be needed, first computing 64 x 32, length 128 ffts, second computing
    128 x 32 length 64 ffts, and finally a kernel computing 128 x 64 length 32 ffts.
    Each of these base radices can futher be divided into factors so that each of these
    base ffts can be computed within one kernel launch using in-register ffts and local
    memory transposes i.e for the first kernel above which computes 64 x 32 ffts on length
    128, 128 can be decomposed into 128 = 16 x 8 i.e. 8 work items can compute 8 length
    16 ffts followed by transpose using local memory followed by each of these eight
    work items computing 2 length 8 ffts thus computing 16 length 8 ffts in total. This
    means only 8 work items are needed for computing one length 128 fft. If we choose
    work group size of say 64, we can compute 64/8 = 8 length 128 ffts within one
    work group. Since we need to compute 64 x 32 length 128 ffts in first kernel, this
    means we need to launch 64 x 32 / 8 = 256 work groups with 64 work items in each
    work group where each work group is computing 8 length 128 ffts where each length
    128 fft is computed by 8 work items. Same logic can be applied to other two kernels
    in this example. Users can play with difference base radices and difference
    decompositions of base radices to generates different kernels and see which gives
    best performance. Following function is just fixed to use 128 as base radix
    """
    base_radix = min(n, 128)

    numR = 0
    N = n
    while N > base_radix:
        N /= base_radix
        numR += 1

    radix = []
    for i in range(numR):
        radix.append(base_radix)

    radix.append(N)
    numR += 1

    R1 = []
    R2 = []
    for i in range(numR):
        B = radix[i]
        if B <= 8:
            R1.append(B)
            R2.append(1)
        else:
            r1 = 2
            r2 = B / r1
            while r2 > r1:
                r1 *= 2
                r2 = B / r1

            R1.append(r1)
            R2.append(r2)

    # sanity checks:
    for r, r1, r2 in zip(radix, R1, R2):
        assert r2 <= r1
        assert r1 * r2 == r
        assert r1 <= MAX_RADIX

    return radix, R1, R2


def get_padding(threads_per_xform, Nprev, threads_req, xforms_per_workgroup, Nr, num_banks):

    if threads_per_xform <= Nprev or Nprev >= num_banks:
        offset = 0
    else:
        numRowsReq = (threads_per_xform if threads_per_xform < num_banks else num_banks) / Nprev
        numColsReq = 1
        if numRowsReq > Nr:
            numColsReq = numRowsReq / Nr
        numColsReq = Nprev * numColsReq
        offset = numColsReq

    if threads_per_xform >= num_banks or xforms_per_workgroup == 1:
        midPad = 0
    else:
        bankNum = ((threads_req + offset) * Nr) & (num_banks - 1)
        if bankNum >= threads_per_xform:
            midPad = 0
        else:
            # TODO: find out which conditions are necessary to execute this code
            midPad = threads_per_xform - bankNum

    lmem_size = (threads_req + offset) * Nr * xforms_per_workgroup + midPad * (xforms_per_workgroup - 1)
    return lmem_size, offset, midPad


def get_local_memory_size(n, radix_array, threads_per_xform, xforms_per_workgroup,
        num_local_mem_banks, min_mem_coalesce_width):

    lmem_size = 0

    # from insertGlobal(Loads/Stores)AndTranspose
    if threads_per_xform < min_mem_coalesce_width:
        lmem_size = max(lmem_size, (n + threads_per_xform) * xforms_per_workgroup)

    Nprev = 1
    len_ = n
    numRadix = len(radix_array)
    for r in range(numRadix):

        numIter = radix_array[0] / radix_array[r]
        threads_req = n / radix_array[r]
        Ncurr = Nprev * radix_array[r]

        if r < numRadix - 1:
            lmem_size_new, offset, midPad = get_padding(
                threads_per_xform, Nprev, threads_req, xforms_per_workgroup,
                radix_array[r], num_local_mem_banks)
            lmem_size = max(lmem_size, lmem_size_new)
            Nprev = Ncurr
            len_ = len_ / radix_array[r]

    return lmem_size


class _FFTKernel:
    """Base class for FFT kernels. Handles compilation and execution."""

    def __init__(self, basis, device_params):
        self._basis = basis
        self._device_params = device_params
        self._normalize = False

    def enable_normalization(self):
        self._normalize = True

    def get_normalization_coeff(self):
        return product([self._basis.shape[i] for i in self._basis.axes])

    def get_batch(self):
        return product(self._basis.shape[:self._axis])

    def prepare_for(self, max_local_size):
        local_size, workgroups_num, kwds = self._generate(max_local_size)

        kwds.update(dict(
            min_mem_coalesce_width=self._device_params.min_mem_coalesce_width[self._basis.dtype.itemsize],
            local_mem_banks=self._device_params.local_mem_banks,
            get_padding=get_padding,
            normalize=self._normalize,
            norm_coeff=self.get_normalization_coeff(),
            wrap_const=lambda x: dtypes.c_constant(x, dtypes.real_for(self._basis.dtype))))

        local_size = local_size
        global_size = local_size * workgroups_num

        return global_size, local_size, kwds


class LocalFFTKernel(_FFTKernel):
    """Generator for 'local' FFT in shared memory"""

    def __init__(self, basis, device_params, n):
        _FFTKernel.__init__(self, basis, device_params)
        self._n = n
        self._axis = len(basis.shape) - 1
        self.name = "fft_local"
        self.inplace_possible = True

    def _generate(self, max_local_size):
        n = self._n

        radix_array = get_radix_array(n)
        if n / radix_array[0] > max_local_size:
            radix_array = get_radix_array(n, use_max_radix=True)

        threads_per_xform = n / radix_array[0]
        local_size = max(64, threads_per_xform)
        if local_size > max_local_size:
            raise OutOfResourcesError
        xforms_per_workgroup = local_size / threads_per_xform
        workgroups_num = min_blocks(self.get_batch(), xforms_per_workgroup)

        lmem_size = get_local_memory_size(
            n, radix_array, threads_per_xform, xforms_per_workgroup,
            self._device_params.local_mem_banks,
            self._device_params.min_mem_coalesce_width[self._basis.dtype.itemsize])

        if lmem_size * self._basis.dtype.itemsize / 2 > self._device_params.local_mem_size:
            raise OutOfResourcesError

        kwds = dict(
            n=n, radix_arr=radix_array,
            lmem_size=lmem_size, threads_per_xform=threads_per_xform,
            xforms_per_workgroup=xforms_per_workgroup,
            global_batch=self.get_batch())

        return local_size, workgroups_num, kwds


class GlobalFFTKernel(_FFTKernel):
    """Generator for 'global' FFT kernel chain."""

    def __init__(self, basis, device_params, pass_num,
            n, curr_n, horiz_wgs, axis, local_batch):

        _FFTKernel.__init__(self, basis, device_params)
        self._n = n
        self._curr_n = curr_n
        self._horiz_wgs = horiz_wgs
        self._axis = axis
        self._local_batch = local_batch
        self._pass_num = pass_num
        self.name = 'fft_global'

        num_passes = len(get_global_radix_info(self._n)[0])
        if self._pass_num == num_passes - 1 and num_passes % 2 == 1:
            self.in_place_possible = True
        else:
            self.in_place_possible = False

    def _generate(self, max_local_size):

        vertical = not (self._axis == len(self._basis.shape) - 1)
        radix_init = self._horiz_wgs if vertical else 1

        radix_arr, radix1_arr, radix2_arr = get_global_radix_info(self._n)
        radix = radix_arr[self._pass_num]
        radix1 = radix1_arr[self._pass_num]
        radix2 = radix2_arr[self._pass_num]
        num_passes = len(radix_arr)

        stride_in = radix_init * product(
            [x for i, x in enumerate(radix_arr) if i != self._pass_num])
        stride_out = radix_init * product(radix_arr[:self._pass_num])
        stride = radix * radix_init * product(radix_arr[:self._pass_num])

        threads_per_xform = radix2

        local_batch = max_local_size if radix2 == 1 else self._local_batch
        local_batch = min(local_batch, stride_in)
        local_size = min(local_batch * threads_per_xform, max_local_size)
        local_batch = local_size / threads_per_xform

        numIter = radix1 / radix2

        workgroups_num = stride_in / local_batch * self.get_batch()
        if not vertical:
            workgroups_num *= self._horiz_wgs

        if radix2 == 1:
            lmem_size = 0
        else:
            if stride_out == 1:
                lmem_size = (radix + 1) * local_batch
            else:
                lmem_size = local_size * radix1

        if lmem_size * self._basis.dtype.itemsize / 2 > self._device_params.local_mem_size:
            raise OutOfResourcesError

        kwds = dict(
            n=self._n, curr_n=self._curr_n, pass_num=self._pass_num,
            lmem_size=lmem_size, local_batch=local_batch, local_size=local_size,
            horiz_wgs=self._horiz_wgs, vertical=vertical,
            radix_arr=radix_arr, radix1_arr=radix1_arr, radix2_arr=radix2_arr,
            radix1=radix1, radix2=radix2, radix=radix,
            stride_in=stride_in, stride_out=stride_out, stride=stride,
            last_pass=(self._pass_num == num_passes - 1))

        return local_size, workgroups_num, kwds

    @staticmethod
    def createChain(basis, device_params, n, horiz_wgs, axis):

        vertical = not (axis == len(basis.shape) - 1)
        coalesce_width = device_params.min_mem_coalesce_width[basis.dtype.itemsize]
        local_batch = min(horiz_wgs, coalesce_width) if vertical else coalesce_width

        radix_arr, _, _ = get_global_radix_info(n)

        curr_n = n
        kernels = []
        for pass_num in range(len(radix_arr)):
            kernels.append(GlobalFFTKernel(
                basis, device_params, pass_num, n, curr_n, horiz_wgs, axis, local_batch))
            curr_n /= radix_arr[pass_num]

        return kernels


def get_fft_1d_kernels(basis, device_params, axis, local_kernel_limit):
    """Create and compile kernels for one of the dimensions"""

    kernels = []

    # TODO: calculate this properly
    max_lmem_fft_size = 1024 if dtypes.is_double(basis.dtype) else 2048

    if axis == len(basis.shape) - 1:
        x = basis.shape[-1]
        if x > max_lmem_fft_size:
            kernels.extend(GlobalFFTKernel.createChain(basis, device_params,
                x, 1, axis))
        elif x > 1:
            if x / MAX_RADIX <= local_kernel_limit:
                kernels.append(LocalFFTKernel(basis, device_params, x))
            else:
                kernels.extend(GlobalFFTKernel.createChain(basis, device_params,
                    x, 1, axis))
    else:
        l = basis.shape[axis]
        if l > 1:
            kernels.extend(GlobalFFTKernel.createChain(
                basis, device_params, l,
                product(basis.shape[axis+1:]), axis))

    return kernels


def get_fft_kernels(basis, device_params, local_kernel_limit):
    kernels = []
    for i, axis in enumerate(reversed(basis.axes)):
        kernels.extend(get_fft_1d_kernels(basis, device_params, axis, local_kernel_limit))
    kernels[-1].enable_normalization()

    return kernels


class FFT(Computation):

    def _get_argnames(self):
        return ('output',), ('input',), ('direction',)

    def _get_basis_for(self, output, input, direction,
            normalize=True, axes=None, support_inplace=False):
        bs = AttrDict(normalize=normalize)

        assert output.shape == input.shape
        assert output.dtype == input.dtype
        #assert len(output.shape) <= 3

        if axes is None:
            axes = tuple(range(len(output.shape)))
        else:
            axes = tuple(axes)

        bs.axes = axes
        bs.shape = output.shape
        bs.dtype = output.dtype
        bs.support_inplace = support_inplace

        return bs

    def _get_argvalues(self, basis):
        return dict(
            output=ArrayValue(basis.shape, basis.dtype),
            input=ArrayValue(basis.shape, basis.dtype),
            direction=ScalarValue(numpy.int32))

    def _construct_operations(self, basis, device_params):

        # While resource consumption of GlobalFFTKernel can be made lower by passing
        # lower value to prepare_for(), LocalFFTKernel may have to be split into several kernels.
        # Therefore, if GlobalFFTKernel.prepare_for() raises OutOfResourcesError,
        # we just call prepare_for() with lower limit, but if LocalFFTKernel.prepare_for()
        # does that, we have to recreate the whole chain.
        local_kernel_limit = device_params.max_work_group_size
        kernel_calls = []

        while local_kernel_limit >= 1:
            # Starting from scratch.
            operations = self._get_operation_recorder()
            kernels = get_fft_kernels(basis, device_params, local_kernel_limit)

            # dumb algorithm using a lot of temporary memory
            temp_names = ['_temp_buffer1', '_temp_buffer2']
            curr_temp = 0
            operations.add_allocation(temp_names[0], product(basis.shape), basis.dtype)
            operations.add_allocation(temp_names[1], product(basis.shape), basis.dtype)

            for i, kernel in enumerate(kernels):
                mem_in = 'input' if i == 0 else mem_out
                mem_out = 'output' if i == len(kernels) - 1 else temp_names[curr_temp]
                curr_temp = 1 - curr_temp
                argnames = [mem_out, mem_in, 'direction']

                # Try to find local size for each of the kernels
                local_size = device_params.max_work_group_size
                local_kernel_fail = False # marks the event when LocalFFTKernel is out of resources
                while local_size >= 1 and not local_kernel_fail:
                    try:
                        gs, ls, kwds = kernel.prepare_for(local_size)
                        operations.add_kernel(
                            TEMPLATE, kernel.name, argnames,
                            global_size=gs, local_size=ls, render_kwds=kwds)
                    except OutOfResourcesError:
                        if isinstance(kernel, GlobalFFTKernel):
                            local_size /= 2
                        else:
                            local_kernel_fail = True
                        continue

                    kernel_calls.append((kernel.name, argnames, gs, ls, kwds))
                    break
                else:
                    raise ValueError(
                        "Could not find suitable call parameters for one of the global kernels")

                if local_kernel_fail:
                    break
            else:
                # everything went well, returning list of calls
                return operations

            # The cycle above received 'break', meaning that LocalFFTKernel was out of resources.
            # Reduce the limit and try to create operations from scratch again.
            local_kernel_limit /= 2

        else:
            raise ValueError("Could not find suitable call parameters for one of the local kernels")

        """
        temp_buffer_needed = False
        for kernel in self._kernels:
            if not kernel.inplace_possible:
                temp_buffer_needed = True

        # If calculations are executed in-place, transformations can get in the way.
        # So all intermediate results are stored in the output buffer using the special view
        operations.add_view('_output_view', '_output', product(basis.shape), basis.dtype)

        # at least one external dram shuffle (transpose) required
        if temp_buffer_needed:
            inplace_done = False
            num_kernels_is_odd = (len(self._kernels) % 2 == 1)

            #mem_objs = ('input', 'output', '_temp_buffer')
            #curr_read  = 0
            #curr_write = 1

            operations.add_allocation('_temp_buffer', product(basis.shape), basis.dtype)
            #buffer_size = self._params.size * batch * self._params.scalar_nbytes
            #self._tempmemobj = self._context.allocate(buffer_size * 2)

            # in-place transform
            if basis.support_inplace:
                curr_read =
                curr_write = 2
                inplace_done = False
            else:
                curr_write = 1 if num_kernels_is_odd else 2

            for i, kernel in enumerate(self._kernels):
                mem_in = 'input' if i == 0 else mem_out

                if i == len(self._kernels) - 1:
                    mem_out = 'output'
                else:


                if is_inplace and num_kernels_is_odd and not inplace_done and kernel.in_place_possible:
                    curr_write = curr_read
                    inplace_done = True

                kernel.prepare(batch)

                self._context.enqueue(kernel, inverse, mem_objs[curr_read], mem_objs[curr_write])

                curr_read  = 1 if (curr_write == 1) else 2
                curr_write = 2 if (curr_write == 1) else 1

        # no dram shuffle (transpose required) transform
        # all kernels can execute in-place.
        else:
            for i, kernel in enumerate(self._kernels):
                mem_in = 'input' if i == 0 else '_output_view'
                mem_out = 'output' if i == len(self.kernels) - 1 else '_output_view'

                #self._context.enqueue(kernel, inverse, mem_objs[curr_read], mem_objs[curr_write])
                operations.add_kernel(kernel,
                    [mem_out, mem_in, 'direction'])
        """
