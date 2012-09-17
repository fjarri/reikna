import numpy

from tigger.helpers import *
from tigger.core import *
import tigger.cluda.dtypes as dtypes
from tigger.cluda import OutOfResourcesError

TEMPLATE = template_for(__file__)


X_DIRECTION = 0
Y_DIRECTION = 1
Z_DIRECTION = 2
MAX_RADIX = 16


def getRadixArray(n, max_radix):
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
    if max_radix > 1:
        max_radix = min(n, max_radix)
        radix_array = []
        while n > max_radix:
            radix_array.append(max_radix)
            n /= max_radix
        radix_array.append(n)
        return radix_array

    if n in [2, 4, 8]:
        return [n]
    elif n in [16, 32, 64]:
        return [8, n / 8]
    elif n == 128:
        return [8, 4, 4]
    elif n == 256:
        return [4, 4, 4, 4]
    elif n == 512:
        return [8, 8, 8]
    elif n == 1024:
        return [16, 16, 4]
    elif n == 2048:
        return [8, 8, 8, 4]
    else:
        raise Exception("Wrong problem size: " + str(n))

def getGlobalRadixInfo(n):
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

    return radix, R1, R2

def getPadding(threads_per_xform, Nprev, threads_req, xforms_per_block, Nr, num_banks):

    if threads_per_xform <= Nprev or Nprev >= num_banks:
        offset = 0
    else:
        numRowsReq = (threads_per_xform if threads_per_xform < num_banks else num_banks) / Nprev
        numColsReq = 1
        if numRowsReq > Nr:
            numColsReq = numRowsReq / Nr
        numColsReq = Nprev * numColsReq
        offset = numColsReq

    if threads_per_xform >= num_banks or xforms_per_block == 1:
        midPad = 0
    else:
        bankNum = ((threads_req + offset) * Nr) & (num_banks - 1)
        if bankNum >= threads_per_xform:
            midPad = 0
        else:
            # TODO: find out which conditions are necessary to execute this code
            midPad = threads_per_xform - bankNum

    smem_size = (threads_req + offset) * Nr * xforms_per_block + midPad * (xforms_per_block - 1)
    return smem_size, offset, midPad

def getSharedMemorySize(n, radix_array, threads_per_xform, xforms_per_block, num_local_mem_banks, min_mem_coalesce_width):

    smem_size = 0

    # from insertGlobal(Loads/Stores)AndTranspose
    if threads_per_xform < min_mem_coalesce_width:
        smem_size = max(smem_size, (n + threads_per_xform) * xforms_per_block)

    Nprev = 1
    len_ = n
    numRadix = len(radix_array)
    for r in range(numRadix):

        numIter = radix_array[0] / radix_array[r]
        threads_req = n / radix_array[r]
        Ncurr = Nprev * radix_array[r]

        if r < numRadix - 1:
            smem_size_new, offset, midPad = getPadding(threads_per_xform, Nprev, threads_req, xforms_per_block,
                radix_array[r], num_local_mem_banks)
            smem_size = max(smem_size, smem_size_new)
            Nprev = Ncurr
            len_ = len_ / radix_array[r]

    return smem_size


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
        axis_nums = {X_DIRECTION: -1, Y_DIRECTION:-2, Z_DIRECTION:-3}
        return product(self._basis.shape[:axis_nums[self._dir]])

    def prepare_for(self, max_local_size):
        self._generate(max_local_size)

        blocks_num = self._blocks_num
        xforms_per_block = self._xforms_per_block

        batch = self.get_batch()
        if self._dir == X_DIRECTION:
            blocks_num = min_blocks(batch, xforms_per_block) * self._blocks_num
        elif self._dir == Y_DIRECTION:
            blocks_num = self._blocks_num * batch
        elif self._dir == Z_DIRECTION:
            blocks_num = self._blocks_num * batch

        local_size = self._block_size
        global_size = local_size * blocks_num

        return global_size, local_size, self._kwds


class LocalFFTKernel(_FFTKernel):
    """Generator for 'local' FFT in shared memory"""

    def __init__(self, basis, device_params, n):
        _FFTKernel.__init__(self, basis, device_params)
        self._n = n
        self.name = "fft_local"

    def _generate(self, max_block_size):
        n = self._n
        assert n <= max_block_size * MAX_RADIX, \
            "Signal length is too big for shared mem fft"

        radix_array = getRadixArray(n, 0)
        if n / radix_array[0] > max_block_size:
            radix_array = getRadixArray(n, MAX_RADIX)

        assert radix_array[0] <= MAX_RADIX, "Max radix choosen is greater than allowed"
        assert n / radix_array[0] <= max_block_size, \
            "Required number of threads per xform greater than maximum block size for local mem fft"

        self._dir = X_DIRECTION
        self.inplace_possible = True

        threads_per_xform = n / radix_array[0]
        block_size = 64 if threads_per_xform <= 64 else threads_per_xform
        if block_size > max_block_size:
            raise OutOfResourcesError
        xforms_per_block = block_size / threads_per_xform
        self._blocks_num = 1
        self._xforms_per_block = xforms_per_block
        self._block_size = block_size

        lmem_size = getSharedMemorySize(n, radix_array, threads_per_xform, xforms_per_block,
            self._device_params.local_mem_banks,
            self._device_params.min_mem_coalesce_width[self._basis.dtype.itemsize])

        if lmem_size * self._basis.dtype.itemsize / 2 > self._device_params.local_mem_size:
            raise OutOfResourcesError

        self._kwds = dict(
            n=n, radix_arr=radix_array,
            lmem_size=lmem_size, threads_per_xform=threads_per_xform,
            xforms_per_block=xforms_per_block,
            shared_mem=lmem_size,
            min_mem_coalesce_width=self._device_params.min_mem_coalesce_width[self._basis.dtype.itemsize],
            num_smem_banks=self._device_params.local_mem_banks,
            log2=log2, getPadding=getPadding,
            normalize=self._normalize,
            norm_coeff=self.get_normalization_coeff(),
            wrap_const=lambda x: dtypes.c_constant(x, dtypes.real_for(self._basis.dtype)),
            global_batch=self.get_batch())


class GlobalFFTKernel(_FFTKernel):
    """Generator for 'global' FFT kernel chain."""

    def __init__(self, basis, device_params, pass_num,
            n, curr_n, horiz_bs, dir, vert_bs, batch_size):
        _FFTKernel.__init__(self, basis, device_params)
        self._n = n
        self._curr_n = curr_n
        self._horiz_bs = horiz_bs
        self._dir = dir
        self._vert_bs = vert_bs
        self._starting_batch_size = batch_size
        self._pass_num = pass_num
        self.name = 'fft_global'

    def _generate(self, max_block_size):

        batch_size = self._starting_batch_size

        vertical = False if self._dir == X_DIRECTION else True

        radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(self._n)

        num_passes = len(radix_arr)

        radix_init = self._horiz_bs if vertical else 1

        radix = radix_arr[self._pass_num]
        radix1 = radix1_arr[self._pass_num]
        radix2 = radix2_arr[self._pass_num]

        stride_in = radix_init
        for i in range(num_passes):
            if i != self._pass_num:
                stride_in *= radix_arr[i]

        stride_out = radix_init
        for i in range(self._pass_num):
            stride_out *= radix_arr[i]

        threads_per_xform = radix2
        batch_size = max_block_size if radix2 == 1 else batch_size
        batch_size = min(batch_size, stride_in)
        self._block_size = batch_size * threads_per_xform
        self._block_size = min(self._block_size, max_block_size)
        batch_size = self._block_size / threads_per_xform
        assert radix2 <= radix1
        assert radix1 * radix2 == radix
        assert radix1 <= MAX_RADIX

        numIter = radix1 / radix2

        blocks_per_xform = stride_in / batch_size
        num_blocks = blocks_per_xform
        if not vertical:
            num_blocks *= self._horiz_bs
        else:
            num_blocks *= self._vert_bs

        if radix2 == 1:
            smem_size = 0
        else:
            if stride_out == 1:
                smem_size = (radix + 1) * batch_size
            else:
                smem_size = self._block_size * radix1

        if smem_size * self._basis.dtype.itemsize / 2 > self._device_params.local_mem_size:
            raise OutOfResourcesError

        self._blocks_num = num_blocks
        self._xforms_per_block = 1

        if self._pass_num == num_passes - 1 and num_passes % 2 == 1:
            self.in_place_possible = True
        else:
            self.in_place_possible = False

        self._kwds = dict(
            n=self._n, curr_n=self._curr_n, pass_num=self._pass_num,
            shared_mem=smem_size, batch_size=batch_size,
            horiz_bs=self._horiz_bs, vert_bs=self._vert_bs, vertical=vertical,
            max_block_size=max_block_size,
            log2=log2, getGlobalRadixInfo=getGlobalRadixInfo,
            normalize=self._normalize,
            norm_coeff=self.get_normalization_coeff(),
            wrap_const=lambda x: dtypes.c_constant(x, dtypes.real_for(self._basis.dtype)),
            global_batch=self.get_batch())

    @staticmethod
    def createChain(basis, device_params, n, horiz_bs, dir, vert_bs):

        batch_size = device_params.min_mem_coalesce_width[basis.dtype.itemsize]
        vertical = not dir == X_DIRECTION

        radix_arr, radix1_arr, radix2_arr = getGlobalRadixInfo(n)

        num_passes = len(radix_arr)

        curr_n = n
        batch_size = min(horiz_bs, batch_size) if vertical else batch_size

        kernels = []

        for pass_num in range(num_passes):
            kernel = GlobalFFTKernel(
                basis, device_params, pass_num, n, curr_n, horiz_bs, dir, vert_bs, batch_size)

            # FIXME: Commented to avoid connection between kernels.
            # Seems to work fine this way too.
            #kernel.compile(fft_params.max_block_size)
            #batch_size = kernel.batch_size

            curr_n /= radix_arr[pass_num]

            kernels.append(kernel)

        return kernels


def get_fft_1d_kernels(basis, device_params, axis):
    """Create and compile kernels for one of the dimensions"""

    kernels = []

    # TODO: calculate this properly
    max_smem_fft_size = 1024 if dtypes.is_double(basis.dtype) else 2048

    if axis == len(basis.shape) - 1:
        x = basis.shape[-1]
        if x > max_smem_fft_size:
            kernels.extend(GlobalFFTKernel.createChain(basis, device_params,
                x, 1, X_DIRECTION, 1))
        elif x > 1:
            radix_array = getRadixArray(x, 0)
            if x / radix_array[0] <= device_params.max_work_group_size:
                kernels.append(LocalFFTKernel(basis, device_params, x))
                #kernel.compile(self._params.max_block_size)
                #kernels.append(kernel)
            else:
                radix_array = getRadixArray(x, MAX_RADIX)
                # TODO: danger - what if during preparation it turns out that
                # maximum work group size for this kernel should be smaller?
                if x / radix_array[0] <= device_params.max_work_group_size:
                    kernels.append(LocalFFTKernel(basis, device_params, x))
                    #kernel.compile(self._params.max_block_size)
                    #kernels.append(kernel)
                else:
                    # TODO: find out which conditions are necessary to execute this code
                    kernels.extend(GlobalFFTKernel.createChain(basis, device_params,
                        x, 1 , X_DIRECTION, 1))
    elif axis == len(basis.shape) - 2:
        y = basis.shape[-2]
        if y > 1:
            kernels.extend(GlobalFFTKernel.createChain(
                basis, device_params, y, basis.shape[-1], Y_DIRECTION, 1))
    elif axis == len(basis.shape) - 3:
        z = basis.shape[-3]
        if z > 1:
            kernels.extend(GlobalFFTKernel.createChain(
                basis, device_params, z,
                basis.shape[-1] * basis.shape[-2], Z_DIRECTION, 1))
    else:
        raise ValueError("Unsupported axis number")

    return kernels


def get_fft_kernels(basis, device_params):
    kernels = []
    for i, axis in enumerate(reversed(basis.axes)):
        kernels.extend(get_fft_1d_kernels(basis, device_params, axis))
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

        assert axes == tuple(range(len(output.shape) - len(axes), len(output.shape)))

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

    def _construct_operations(self, operations, basis, device_params):

        kernels = get_fft_kernels(basis, device_params)

        # dumb algorithm using a lot of temporary memory
        temp_names = ['_temp_buffer1', '_temp_buffer2']
        curr_temp = 0
        operations.add_allocation(temp_names[0], product(basis.shape), basis.dtype)
        operations.add_allocation(temp_names[1], product(basis.shape), basis.dtype)
        for i, kernel in enumerate(kernels):
            mem_in = 'input' if i == 0 else mem_out
            mem_out = 'output' if i == len(kernels) - 1 else temp_names[curr_temp]
            curr_temp = 1 - curr_temp

            local_size = device_params.max_work_group_size
            while local_size >= 1:
                try:
                    gs, ls, kwds = kernel.prepare_for(local_size)
                    operations.add_kernel(
                        TEMPLATE, kernel.name,
                        [mem_out, mem_in, 'direction'],
                        global_size=gs, local_size=ls, render_kwds=kwds)
                except OutOfResourcesError:
                    local_size /= 2
                    continue

                break

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

