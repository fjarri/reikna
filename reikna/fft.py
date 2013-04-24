import numpy

from reikna.helpers import *
from reikna.core import *
from reikna.cluda import Module
from reikna.cluda import functions
import reikna.cluda.dtypes as dtypes
from reikna.cluda import OutOfResourcesError
from reikna.elementwise import specialize_elementwise

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
    if n != 2 ** log2(n):
        raise ValueError("Wrong problem size: " + str(n))

    if use_max_radix:
        radix = min(n, MAX_RADIX)
        radix_array = []
        while n > radix:
            radix_array.append(radix)
            n //= radix
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
            # Naive algorithm, can be imroved.
            l = log2(n)
            num_elems = min_blocks(l, 4)
            return [16] * (num_elems - 1) + [16 if l % 4 == 0 else 2 ** (l % 4)]


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
        N //= base_radix
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
            r2 = B // r1
            while r2 > r1:
                r1 *= 2
                r2 = B // r1

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
        numRowsReq = (threads_per_xform if threads_per_xform < num_banks else num_banks) // Nprev
        numColsReq = 1
        if numRowsReq > Nr:
            numColsReq = numRowsReq // Nr
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

        threads_req = n // radix_array[r]
        Ncurr = Nprev * radix_array[r]

        if r < numRadix - 1:
            lmem_size_new, offset, midPad = get_padding(
                threads_per_xform, Nprev, threads_req, xforms_per_workgroup,
                radix_array[r], num_local_mem_banks)
            lmem_size = max(lmem_size, lmem_size_new)
            Nprev = Ncurr
            len_ = len_ // radix_array[r]

    return lmem_size


def get_kweights(size_real, size_bound):
    """
    Returns weights to be applied as a part of Bluestein's algorithm
    between forward and inverse FFTs.
    """

    args = lambda ns: 1j * numpy.pi / size_real * ns ** 2

    n_v = numpy.concatenate([
        numpy.arange(size_bound - size_real +1),
        numpy.arange(size_real - 1, 0, -1)])

    return numpy.concatenate([
        numpy.fft.fft(numpy.exp(args(n_v))),
        numpy.fft.ifft(numpy.exp(-args(n_v))) * size_bound / size_real])


class _FFTKernel:
    """Base class for FFT kernels. Handles compilation and execution."""

    def __init__(self, basis, device_params):
        self._basis = basis
        self._device_params = device_params
        self._normalize = False
        self.kweights = None

    def enable_normalization(self):
        self._normalize = True

    def prepare_for(self, max_local_size):
        local_size, workgroups_num, kwds = self._generate(max_local_size)
        basis = self._basis

        kwds.update(dict(
            min_mem_coalesce_width=self._device_params.min_mem_coalesce_width[basis.dtype.itemsize],
            local_mem_banks=self._device_params.local_mem_banks,
            get_padding=get_padding,
            normalize=self._normalize,
            wrap_const=lambda x: dtypes.c_constant(x, dtypes.real_for(basis.dtype)),
            min_blocks=min_blocks,
            takes_kweights=(self.kweights is not None),
            pad_in=(self._fft_size != self._fft_size_real and self._pass_num == 0
                and not self._reverse_direction),
            unpad_out=(self._fft_size != self._fft_size_real and self._last_pass
                and self._reverse_direction),
            reverse_direction=self._reverse_direction,
            mul=functions.mul(basis.dtype, basis.dtype),
            polar=functions.polar(dtypes.real_for(basis.dtype)),
            cdivs=functions.div(basis.dtype, dtypes.real_for(basis.dtype))))

        local_size = local_size
        global_size = local_size * workgroups_num

        return global_size, local_size, kwds


class LocalFFTKernel(_FFTKernel):
    """Generator for 'local' FFT in shared memory"""

    def __init__(self, basis, device_params, outer_batch, fft_size, fft_size_real,
            reverse_direction):
        _FFTKernel.__init__(self, basis, device_params)
        self._fft_size = fft_size
        self._outer_batch = outer_batch
        self.name = "fft_local"
        self.inplace_possible = True
        self._reverse_direction = reverse_direction
        self._fft_size_real = fft_size_real

        self._pass_num = 0
        self._last_pass = True

        if reverse_direction:
            self.output_shape = (outer_batch, fft_size_real)
        else:
            self.output_shape = (outer_batch, fft_size)

        if fft_size_real != fft_size and reverse_direction:
            self.kweights = get_kweights(fft_size_real, fft_size)

        self.enable_normalization()

    def _generate(self, max_local_size):
        fft_size = self._fft_size

        radix_array = get_radix_array(fft_size)
        if fft_size // radix_array[0] > max_local_size:
            radix_array = get_radix_array(fft_size, use_max_radix=True)

        threads_per_xform = fft_size // radix_array[0]
        local_size = max(64, threads_per_xform)
        if local_size > max_local_size:
            raise OutOfResourcesError
        xforms_per_workgroup = local_size // threads_per_xform
        workgroups_num = min_blocks(self._outer_batch, xforms_per_workgroup)

        lmem_size = get_local_memory_size(
            fft_size, radix_array, threads_per_xform, xforms_per_workgroup,
            self._device_params.local_mem_banks,
            self._device_params.min_mem_coalesce_width[self._basis.dtype.itemsize])

        if lmem_size * self._basis.dtype.itemsize // 2 > self._device_params.local_mem_size:
            raise OutOfResourcesError

        kwds = dict(
            fft_size=fft_size, fft_size_real=self._fft_size_real, radix_arr=radix_array,
            lmem_size=lmem_size, threads_per_xform=threads_per_xform,
            xforms_per_workgroup=xforms_per_workgroup,
            outer_batch=self._outer_batch)

        return local_size, workgroups_num, kwds


class GlobalFFTKernel(_FFTKernel):
    """Generator for 'global' FFT kernel chain."""

    def __init__(self, basis, device_params, outer_batch, fft_size, curr_size,
            fft_size_real, inner_batch, pass_num, reverse_direction):

        _FFTKernel.__init__(self, basis, device_params)
        self._fft_size = fft_size
        self._curr_size = curr_size
        self._inner_batch = inner_batch
        self._outer_batch = outer_batch
        self._pass_num = pass_num
        self.name = 'fft_global'

        self._reverse_direction = reverse_direction
        self._fft_size_real = fft_size_real

        num_passes = len(get_global_radix_info(fft_size)[0])
        if self._pass_num == num_passes - 1 and num_passes % 2 == 1:
            self.inplace_possible = True
        else:
            self.inplace_possible = False

        if pass_num == 0 and reverse_direction:
            self.kweights = get_kweights(fft_size_real, fft_size)

        self._last_pass = (pass_num == num_passes - 1)
        if pass_num == num_passes - 1 and reverse_direction:
            self.output_shape = (outer_batch, fft_size_real, inner_batch)
        else:
            self.output_shape = (outer_batch, fft_size, inner_batch)

    def _generate(self, max_local_size):

        radix_arr, radix1_arr, radix2_arr = get_global_radix_info(self._fft_size)

        radix = radix_arr[self._pass_num]
        radix1 = radix1_arr[self._pass_num]
        radix2 = radix2_arr[self._pass_num]
        num_passes = len(radix_arr)

        stride_out = self._inner_batch * product(radix_arr[:self._pass_num])
        stride = stride_out * radix
        stride_in = stride_out * product(radix_arr[self._pass_num+1:])

        threads_per_xform = radix2

        coalesce_width = self._device_params.min_mem_coalesce_width[self._basis.dtype.itemsize]
        local_batch = max_local_size if radix2 == 1 else coalesce_width
        local_batch = min(local_batch, stride_in)
        local_size = min(local_batch * threads_per_xform, max_local_size)
        local_batch = local_size // threads_per_xform

        workgroups_num = min_blocks(stride_in, local_batch) * self._outer_batch

        if radix2 == 1:
            lmem_size = 0
        else:
            if stride_out == 1:
                lmem_size = (radix + 1) * local_batch
            else:
                lmem_size = local_size * radix1

        if lmem_size * self._basis.dtype.itemsize // 2 > self._device_params.local_mem_size:
            raise OutOfResourcesError

        kwds = dict(
            fft_size=self._fft_size, curr_size=self._curr_size, fft_size_real=self._fft_size_real,
            pass_num=self._pass_num,
            lmem_size=lmem_size, local_batch=local_batch, local_size=local_size,
            inner_batch=self._inner_batch,
            radix_arr=radix_arr, radix1_arr=radix1_arr, radix2_arr=radix2_arr,
            radix1=radix1, radix2=radix2, radix=radix,
            stride_in=stride_in, stride_out=stride_out, stride=stride,
            last_pass=(self._pass_num == num_passes - 1))

        return local_size, workgroups_num, kwds

    @staticmethod
    def createChain(basis, device_params, outer_batch, fft_size, fft_size_real, inner_batch,
            reverse_direction):

        radix_arr, _, _ = get_global_radix_info(fft_size)

        curr_size = fft_size
        kernels = []
        for pass_num in range(len(radix_arr)):
            kernels.append(GlobalFFTKernel(
                basis, device_params, outer_batch, fft_size,
                curr_size, fft_size_real, inner_batch, pass_num,
                reverse_direction))
            curr_size //= radix_arr[pass_num]

        kernels[-1].enable_normalization()

        return kernels


def get_fft_1d_kernels(basis, device_params, outer_batch, fft_size, inner_batch,
        local_kernel_limit, reverse_direction=False, fft_size_real=None):
    """Create and compile kernels for one of the dimensions"""

    kernels = []

    if fft_size_real is None:
        fft_size_real = fft_size

    if (inner_batch == 1 and fft_size // MAX_RADIX <= local_kernel_limit):
        kernels.append(LocalFFTKernel(
            basis, device_params, outer_batch, fft_size, fft_size_real,
            reverse_direction))
    else:
        kernels.extend(GlobalFFTKernel.createChain(
            basis, device_params, outer_batch, fft_size, fft_size_real,
            inner_batch, reverse_direction))

    return kernels


def get_fft_kernels(basis, device_params, local_kernel_limit):
    kernels = []
    for i, axis in enumerate(reversed(basis.axes)):
        outer_batch = product(basis.shape[:axis])
        fft_size = basis.shape[axis]
        inner_batch = product(basis.shape[axis+1:])

        if fft_size == 1:
            continue

        bounding_size = bounding_power_of_2(fft_size)

        if bounding_size == fft_size:
            kernels.extend(get_fft_1d_kernels(
                basis, device_params, outer_batch, fft_size,
                inner_batch, local_kernel_limit))
        else:
            # padding FFT for the chirp-z transform
            fft_size_padded = 2 * bounding_size
            args = (basis, device_params, outer_batch, fft_size_padded,
                inner_batch, local_kernel_limit)

            new_kernels = []
            new_kernels.extend(get_fft_1d_kernels(
                *args, fft_size_real=fft_size))
            new_kernels.extend(get_fft_1d_kernels(
                *args, reverse_direction=True, fft_size_real=fft_size))

            # Since during pad-in or pad-out input and output blocks are no longer aligned,
            # these kernels lose their inplace_possible property
            new_kernels[0].inplace_possible = False
            new_kernels[-1].inplace_possible = False

            kernels.extend(new_kernels)

    return kernels


class FFT(Computation):
    """
    Performs the Fast Fourier Transform.
    The interface is similar to :py:class:`numpy.fft.fftn`.
    The inverse transform is normalized so that ``IFFT(FFT(X)) = X``.

    .. py:method:: prepare_for(output, input, direction, axes=None)

        :param output: output array.
        :param input: input array (with the same size as ``output``).
        :param direction: ``-1`` for forward transform, ``1`` for inverse transform.
        :param axes: a tuple with axes over which to perform the transform.
            If not given, the transform is performed over all the axes.

    .. note::
        Current algorithm works most effectively with array dimensions being power of 2
        This mostly applies to the axes over which the transform is performed,
        beacuse otherwise the computation falls back to the Bluestein's algorithm,
        which effectively halves the performance.
    """

    def _get_argnames(self):
        return ('output',), ('input',), ('direction',)

    def _get_basis_for(self, output, input, direction, axes=None):
        bs = AttrDict()

        assert output.shape == input.shape
        assert output.dtype == input.dtype

        if axes is None:
            axes = tuple(range(len(output.shape)))
        else:
            axes = tuple(axes)

        bs.axes = axes
        bs.shape = output.shape
        bs.dtype = output.dtype

        return bs

    def _get_argvalues(self, basis):
        return dict(
            output=ArrayValue(basis.shape, basis.dtype),
            input=ArrayValue(basis.shape, basis.dtype),
            direction=ScalarValue(numpy.int32))

    def _construct_operations(self, basis, device_params):

        if product([basis.shape[i] for i in basis.axes]) == 1:
            # Trivial problem. Need to add a dummy kernel
            # because we still have to run transformations.
            operations = self._get_operation_recorder()

            code = lambda output, input: Module(
                template_def(
                    ['output', 'input'],
                    """
                    ${output.store}(idx, ${input.load}(idx));
                    """),
                snippet=True)

            identity = self.get_nested_computation(
                specialize_elementwise('output', 'input', None, code))
            operations.add_computation(identity, 'output', 'input')
            return operations

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

            for i, kernel in enumerate(kernels):

                mem_in = 'input' if i == 0 else mem_out
                if i == len(kernels) - 1:
                    mem_out = 'output'
                else:
                    mem_out = operations.add_allocation(kernel.output_shape, basis.dtype)

                if kernel.kweights is not None:
                    kweights = operations.add_const_allocation(
                        kernel.kweights.astype(basis.dtype))
                    kweights_arg = [kweights]
                else:
                    kweights_arg = []

                argnames = [mem_out, mem_in] + kweights_arg + ['direction']

                # Try to find local size for each of the kernels
                local_size = device_params.max_work_group_size
                local_kernel_fail = False # marks the event when LocalFFTKernel is out of resources
                while local_size >= 1 and not local_kernel_fail:
                    try:
                        gs, ls, kwds = kernel.prepare_for(local_size)
                        operations.add_kernel(
                            TEMPLATE.get_def(kernel.name), argnames,
                            global_size=gs, local_size=ls, render_kwds=kwds,
                            dependencies=([] if kernel.inplace_possible else [(mem_in, mem_out)]))
                    except OutOfResourcesError:
                        if isinstance(kernel, GlobalFFTKernel):
                            local_size //= 2
                        else:
                            local_kernel_fail = True
                        continue

                    break
                else:
                    if not local_kernel_fail:
                        raise ValueError(
                            "Could not find suitable call parameters for one of the global kernels")

                if local_kernel_fail:
                    break
            else:
                # everything went well, returning list of calls
                return operations

            # The cycle above received 'break', meaning that LocalFFTKernel was out of resources.
            # Reduce the limit and try to create operations from scratch again.
            local_kernel_limit //= 2

        else:
            raise ValueError("Could not find suitable call parameters for one of the local kernels")
