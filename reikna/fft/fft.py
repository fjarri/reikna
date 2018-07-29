import numpy

import reikna.helpers as helpers
from reikna.core import Computation, Parameter, Annotation
from reikna.cluda import functions
import reikna.cluda.dtypes as dtypes
from reikna.cluda import OutOfResourcesError
from reikna.algorithms import PureParallel
from reikna.transformations import copy

TEMPLATE = helpers.template_for(__file__)


MAX_RADIX = 16


def get_radix_array(size, use_max_radix=False):
    """
    For any ``size``, this function decomposes ``size`` into factors for loacal memory tranpose
    based fft. Factors (radices) are sorted such that the first one (radix_array[0])
    is the largest. This base radix determines the number of registers used by each
    work item and product of remaining radices determine the size of work group needed.
    To make things concrete with and example, suppose ``size`` = 1024. It is decomposed into
    1024 = 16 x 16 x 4. Hence kernel uses float2 a[16], for local in-register fft and
    needs 16 x 4 = 64 work items per work group. So kernel first performance 64 length
    16 ffts (64 work items working in parallel) following by transpose using local
    memory followed by again 64 length 16 ffts followed by transpose using local memory
    followed by 256 length 4 ffts. For the last step since with size of work group is
    64 and each work item can array for 16 values, 64 work items can compute 256 length
    4 ffts by each work item computing 4 length 4 ffts.
    Similarly for ``size`` = 2048 = 8 x 8 x 8 x 4, each work group has 8 x 8 x 4 = 256 work
    iterms which each computes 256 (in-parallel) length 8 ffts in-register, followed
    by transpose using local memory, followed by 256 length 8 in-register ffts, followed
    by transpose using local memory, followed by 256 length 8 in-register ffts, followed
    by transpose using local memory, followed by 512 length 4 in-register ffts. Again,
    for the last step, each work item computes two length 4 in-register ffts and thus
    256 work items are needed to compute all 512 ffts.
    For ``size`` = 32 = 8 x 4, 4 work items first compute 4 in-register
    lenth 8 ffts, followed by transpose using local memory followed by 8 in-register
    length 4 ffts, where each work item computes two length 4 ffts thus 4 work items
    can compute 8 length 4 ffts. However if work group size of say 64 is choosen,
    each work group can compute 64/ 4 = 16 size 32 ffts (batched transform).
    Users can play with these parameters to figure what gives best performance on
    their particular device i.e. some device have less register space thus using
    smaller base radix can avoid spilling ... some has small local memory thus
    using smaller work group size may be required etc
    """
    assert size == 2 ** helpers.log2(size)

    if use_max_radix:
        radix = min(size, MAX_RADIX)
        radix_array = []
        while size > radix:
            radix_array.append(radix)
            size //= radix
        radix_array.append(size)
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
        if size in arrays:
            return arrays[size]
        else:
            # Naive algorithm, can be imroved.
            lsize = helpers.log2(size)
            num_elems = helpers.min_blocks(lsize, 4)
            return [16] * (num_elems - 1) + [16 if lsize % 4 == 0 else 2 ** (lsize % 4)]


def get_global_radix_info(size):
    """
    For ``size`` larger than what can be computed using local memory fft, global transposes
    multiple kernel launces is needed. For these sizes, ``size`` can be decomposed using
    much larger base radices i.e. say ``size`` = 262144 = 128 x 64 x 32. Thus three kernel
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
    assert size == 2 ** helpers.log2(size)

    base_radix = min(size, 128)

    num_radices = 0
    while size > base_radix:
        size //= base_radix
        num_radices += 1

    radix_list = [base_radix] * num_radices + [size]
    radix1_list = []
    radix2_list = []

    for radix in radix_list:
        if radix <= 8:
            radix1_list.append(radix)
            radix2_list.append(1)
        else:
            radix1 = 2
            radix2 = radix // radix1
            while radix2 > radix1:
                radix1 *= 2
                radix2 = radix // radix1

            radix1_list.append(radix1)
            radix2_list.append(radix2)

    # sanity checks:
    for radix, radix1, radix2 in zip(radix_list, radix1_list, radix2_list):
        assert radix2 <= radix1
        assert radix1 * radix2 == radix
        assert radix1 <= MAX_RADIX

    return radix_list, radix1_list, radix2_list


def get_padding(threads_per_xform, radix_prev, threads_req, xforms_per_workgroup, radix, num_banks):

    if threads_per_xform <= radix_prev or radix_prev >= num_banks:
        offset = 0
    else:
        num_rows_req = ((threads_per_xform if threads_per_xform < num_banks else num_banks) //
            radix_prev)
        num_cols_req = 1
        if num_rows_req > radix:
            num_cols_req = num_rows_req // radix
        num_cols_req *= radix_prev
        offset = num_cols_req

    if threads_per_xform >= num_banks or xforms_per_workgroup == 1:
        mid_pad = 0
    else:
        bank_num = ((threads_req + offset) * radix) & (num_banks - 1)
        if bank_num >= threads_per_xform:
            mid_pad = 0
        else:
            # TODO: find out which conditions are necessary to execute this code
            mid_pad = threads_per_xform - bank_num

    lmem_size = ((threads_req + offset) * radix * xforms_per_workgroup +
        mid_pad * (xforms_per_workgroup - 1))
    return lmem_size, offset, mid_pad


def get_local_memory_size(size, radix_array, threads_per_xform, xforms_per_workgroup,
        num_local_mem_banks, min_mem_coalesce_width):

    lmem_size = 0

    # from insertGlobal(Loads/Stores)AndTranspose
    if threads_per_xform < min_mem_coalesce_width:
        lmem_size = max(lmem_size, (size + threads_per_xform) * xforms_per_workgroup)

    radix_prev = 1
    len_ = size
    for i, radix in enumerate(radix_array):

        threads_req = size // radix
        radix_curr = radix_prev * radix

        if i < len(radix_array) - 1:
            lmem_size_new, _, _ = get_padding(
                threads_per_xform, radix_prev, threads_req, xforms_per_workgroup,
                radix, num_local_mem_banks)
            lmem_size = max(lmem_size, lmem_size_new)
            radix_prev = radix_curr
            len_ = len_ // radix

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

    return numpy.vstack([
        numpy.fft.fft(numpy.exp(args(n_v))),
        numpy.fft.ifft(numpy.exp(-args(n_v))) * size_bound / size_real])


def get_common_kwds(dtype, device_params):
    return dict(
        dtype=dtype,
        min_mem_coalesce_width=device_params.min_mem_coalesce_width[dtype.itemsize],
        local_mem_banks=device_params.local_mem_banks,
        get_padding=get_padding,
        wrap_const=lambda x: dtypes.c_constant(x, dtypes.real_for(dtype)),
        min_blocks=helpers.min_blocks,
        mul=functions.mul(dtype, dtype),
        polar_unit=functions.polar_unit(dtypes.real_for(dtype)),
        cdivs=functions.div(dtype, numpy.uint32, out_dtype=dtype))


class LocalFFTKernel:
    """Generator for 'local' FFT in shared memory"""

    def __init__(self, dtype, device_params, outer_shape, fft_size, fft_size_real,
            inner_shape, reverse_direction):

        self.name = "fft_local"
        self.inplace_possible = True
        self.output_shape = outer_shape + (fft_size_real if reverse_direction else fft_size,)
        if fft_size_real != fft_size and reverse_direction:
            self.kweights = get_kweights(fft_size_real, fft_size)
        else:
            self.kweights = None

        self._fft_size = fft_size
        self._fft_size_real = fft_size_real
        self._outer_batch = helpers.product(outer_shape)
        self._local_mem_size = device_params.local_mem_size
        self._itemsize = dtype.itemsize

        self._constant_kwds = get_common_kwds(dtype, device_params)
        self._constant_kwds.update(dict(
            takes_kweights=(self.kweights is not None),
            input_slices=(len(outer_shape), 1, len(inner_shape)),
            output_slices=(len(outer_shape), 1, len(inner_shape)),
            pad_in=(fft_size != fft_size_real and not reverse_direction),
            unpad_out=(fft_size != fft_size_real and reverse_direction),
            reverse_direction=reverse_direction,
            normalize=True))

    def prepare_for(self, max_local_size):
        kwds = dict(self._constant_kwds)
        fft_size = self._fft_size

        radix_array = get_radix_array(fft_size)
        if fft_size // radix_array[0] > max_local_size:
            radix_array = get_radix_array(fft_size, use_max_radix=True)

        threads_per_xform = fft_size // radix_array[0]
        local_size = max(64, threads_per_xform)
        if local_size > max_local_size:
            raise OutOfResourcesError
        xforms_per_workgroup = local_size // threads_per_xform
        workgroups_num = helpers.min_blocks(self._outer_batch, xforms_per_workgroup)

        lmem_size = get_local_memory_size(
            fft_size, radix_array, threads_per_xform, xforms_per_workgroup,
            kwds['local_mem_banks'], kwds['min_mem_coalesce_width'])

        if lmem_size * self._itemsize // 2 > self._local_mem_size:
            raise OutOfResourcesError

        kwds.update(dict(
            fft_size=fft_size, fft_size_real=self._fft_size_real, radix_arr=radix_array,
            lmem_size=lmem_size, threads_per_xform=threads_per_xform,
            xforms_per_workgroup=xforms_per_workgroup,
            outer_batch=self._outer_batch))

        return local_size * workgroups_num, local_size, kwds


class GlobalFFTKernel:
    """Generator for 'global' FFT kernel chain."""

    def __init__(self, dtype, device_params, outer_shape, fft_size, curr_size,
            fft_size_real, inner_shape, pass_num, reverse_direction):

        num_passes = len(get_global_radix_info(fft_size)[0])
        real_output_shape = (pass_num == num_passes - 1 and reverse_direction)

        self.name = 'fft_global'
        self.inplace_possible = (pass_num == num_passes - 1 and num_passes % 2 == 1)
        self.output_shape = (outer_shape +
            (fft_size_real if real_output_shape else fft_size,) + inner_shape)
        if fft_size != fft_size_real and pass_num == 0 and reverse_direction:
            self.kweights = get_kweights(fft_size_real, fft_size)
        else:
            self.kweights = None

        self._fft_size = fft_size
        self._curr_size = curr_size
        self._fft_size_real = fft_size_real
        self._local_mem_size = device_params.local_mem_size
        self._itemsize = dtype.itemsize
        self._inner_batch = helpers.product(inner_shape)
        self._outer_batch = helpers.product(outer_shape)
        self._pass_num = pass_num
        self._last_pass = (pass_num == num_passes - 1)

        self._constant_kwds = get_common_kwds(dtype, device_params)
        self._constant_kwds.update(dict(
            takes_kweights=(self.kweights is not None),
            input_slices=(len(outer_shape), 1, len(inner_shape)),
            output_slices=(len(outer_shape), 1, len(inner_shape)),
            pad_in=(fft_size != fft_size_real and pass_num == 0 and not reverse_direction),
            unpad_out=(fft_size != fft_size_real and self._last_pass and reverse_direction),
            reverse_direction=reverse_direction,
            normalize=self._last_pass))

    def prepare_for(self, max_local_size):
        kwds = dict(self._constant_kwds)

        radix_arr, radix1_arr, radix2_arr = get_global_radix_info(self._fft_size)

        radix = radix_arr[self._pass_num]
        radix1 = radix1_arr[self._pass_num]
        radix2 = radix2_arr[self._pass_num]

        stride_out = self._inner_batch * helpers.product(radix_arr[:self._pass_num])
        stride = stride_out * radix
        stride_in = stride_out * helpers.product(radix_arr[self._pass_num+1:])

        threads_per_xform = radix2

        coalesce_width = kwds['min_mem_coalesce_width']
        local_batch = max_local_size if radix2 == 1 else coalesce_width
        local_batch = min(local_batch, stride_in)
        local_size = min(local_batch * threads_per_xform, max_local_size)
        local_batch = local_size // threads_per_xform

        workgroups_num = helpers.min_blocks(stride_in, local_batch) * self._outer_batch

        if radix2 == 1:
            lmem_size = 0
        else:
            if stride_out == 1:
                lmem_size = (radix + 1) * local_batch
            else:
                lmem_size = local_size * radix1

        if lmem_size * self._itemsize // 2 > self._local_mem_size:
            raise OutOfResourcesError

        kwds.update(self._constant_kwds)
        kwds.update(dict(
            fft_size=self._fft_size, curr_size=self._curr_size, fft_size_real=self._fft_size_real,
            pass_num=self._pass_num,
            lmem_size=lmem_size, local_batch=local_batch, local_size=local_size,
            inner_batch=self._inner_batch,
            radix_arr=radix_arr, radix1_arr=radix1_arr, radix2_arr=radix2_arr,
            radix1=radix1, radix2=radix2, radix=radix,
            stride_in=stride_in, stride_out=stride_out, stride=stride,
            last_pass=self._last_pass))

        return workgroups_num * local_size, local_size, kwds

    @staticmethod
    def create_chain(dtype, device_params, outer_shape, fft_size, fft_size_real, inner_shape,
            reverse_direction):

        radix_arr, _, _ = get_global_radix_info(fft_size)

        curr_size = fft_size
        kernels = []
        for pass_num in range(len(radix_arr)):
            kernels.append(GlobalFFTKernel(
                dtype, device_params, outer_shape, fft_size,
                curr_size, fft_size_real, inner_shape, pass_num,
                reverse_direction))
            curr_size //= radix_arr[pass_num]

        return kernels


def get_fft_1d_kernels(dtype, device_params, outer_shape, fft_size, inner_shape,
        local_kernel_limit, reverse_direction=False, fft_size_real=None):
    """Create and compile kernels for one of the dimensions"""

    kernels = []

    if fft_size_real is None:
        fft_size_real = fft_size

    if (helpers.product(inner_shape) == 1 and fft_size // MAX_RADIX <= local_kernel_limit):
        kernels.append(LocalFFTKernel(
            dtype, device_params, outer_shape, fft_size, fft_size_real,
            inner_shape, reverse_direction))
    else:
        kernels.extend(GlobalFFTKernel.create_chain(
            dtype, device_params, outer_shape, fft_size, fft_size_real,
            inner_shape, reverse_direction))

    return kernels


def get_fft_kernels(input_shape, dtype, axes, device_params, local_kernel_limit):
    kernels = []

    # Starting from the most local transformation, for the sake of neatness.
    # Does not really matter.
    for axis in reversed(axes):
        outer_shape = input_shape[:axis]
        fft_size = input_shape[axis]
        inner_shape = input_shape[axis+1:]

        if fft_size == 1:
            continue

        bounding_size = helpers.bounding_power_of_2(fft_size)

        if bounding_size == fft_size:
            kernels.extend(get_fft_1d_kernels(
                dtype, device_params, outer_shape, fft_size,
                inner_shape, local_kernel_limit))
        else:
            # padding FFT for the chirp-z transform
            fft_size_padded = 2 * bounding_size
            args = (dtype, device_params, outer_shape, fft_size_padded,
                inner_shape, local_kernel_limit)

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


class LocalKernelFail(Exception):
    pass

class GlobalKernelFail(Exception):
    pass


class FFT(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    Performs the Fast Fourier Transform.
    The interface is similar to ``numpy.fft.fftn``.
    The inverse transform is normalized so that ``IFFT(FFT(X)) = X``.

    :param arr_t: an array-like defining the problem array.
    :param axes: a tuple with axes over which to perform the transform.
        If not given, the transform is performed over all the axes.

    .. note::
        Current algorithm works most effectively with array dimensions being power of 2
        This mostly applies to the axes over which the transform is performed,
        beacuse otherwise the computation falls back to the Bluestein's algorithm,
        which effectively halves the performance.

    .. py:method:: compiled_signature(output:o, input:i, inverse:s)

        ``output`` and ``input`` may be the same array.

        ..
            The only case where it would matter is when we only have one kernel in the plan,
            and it has ``inplace_possible==False``.
            Local kernels are always inplace, so it must be a global kernel.
            But if there is only one global kernel in the list, it will have its
            ``inplace_possible`` set to ``True`` (see the condition in ``GlobalFFTKernel``).
            Therefore our FFT is always guaranteed to be inplace.

        :param output: an array with the attributes of ``arr_t``.
        :param input: an array with the attributes of ``arr_t``.
        :param inverse: a scalar value castable to integer.
            If ``0``, ``output`` contains the forward FFT of ``input``,
            if ``1``, the inverse one.
    """

    def __init__(self, arr_t, axes=None):

        if not dtypes.is_complex(arr_t.dtype):
            raise ValueError("FFT computation requires array of a complex dtype")

        Computation.__init__(self, [
            Parameter('output', Annotation(arr_t, 'o')),
            Parameter('input', Annotation(arr_t, 'i')),
            Parameter('inverse', Annotation(numpy.int32), default=0)])

        if axes is None:
            axes = tuple(range(len(arr_t.shape)))
        else:
            axes = tuple(axes)
        self._axes = axes

    def _build_trivial_plan(self, plan_factory, output, input_):
        # Trivial problem. Need to add a dummy kernel
        # because we still have to run transformations.

        plan = plan_factory()

        copy_trf = copy(input_, out_arr_t=output)
        copy_comp = PureParallel.from_trf(copy_trf, copy_trf.input)
        plan.computation_call(copy_comp, output, input_)

        return plan

    def _build_limited_plan(self, plan_factory, device_params, local_kernel_limit,
            output, input_, inverse):

        plan = plan_factory()
        kernels = get_fft_kernels(
            input_.shape, input_.dtype, self._axes, device_params, local_kernel_limit)

        mem_out = None
        for i, kernel in enumerate(kernels):

            mem_in = input_ if i == 0 else mem_out
            if i == len(kernels) - 1:
                mem_out = output
            elif kernel.inplace_possible and mem_in is not input_:
                mem_out = mem_in
            else:
                mem_out = plan.temp_array(kernel.output_shape, output.dtype)

            if kernel.kweights is not None:
                kweights = plan.persistent_array(kernel.kweights.astype(output.dtype))
                kweights_arg = [kweights]
            else:
                kweights_arg = []

            argnames = [mem_out, mem_in] + kweights_arg + [inverse]

            # Try to find local size for each of the kernels
            local_size = device_params.max_work_group_size
            while local_size >= 1:
                try:
                    gsize, lsize, kwds = kernel.prepare_for(local_size)
                    plan.kernel_call(
                        TEMPLATE.get_def(kernel.name), argnames,
                        kernel_name="kernel_fft",
                        global_size=gsize, local_size=lsize, render_kwds=kwds)
                except OutOfResourcesError:
                    if isinstance(kernel, GlobalFFTKernel):
                        local_size //= 2
                        continue
                    else:
                        raise LocalKernelFail
                break
            else:
                raise GlobalKernelFail

        return plan

    def _build_plan(self, plan_factory, device_params, output, input_, inverse):

        if helpers.product([input_.shape[i] for i in self._axes]) == 1:
            return self._build_trivial_plan(plan_factory, output, input_)

        # While resource consumption of GlobalFFTKernel can be made lower by passing
        # lower value to prepare_for(), LocalFFTKernel may have to be split into several kernels.
        # Therefore, if GlobalFFTKernel.prepare_for() raises OutOfResourcesError,
        # we just call prepare_for() with lower limit, but if LocalFFTKernel.prepare_for()
        # does that, we have to recreate the whole chain.
        local_kernel_limit = device_params.max_work_group_size

        while local_kernel_limit >= 1:
            try:
                plan = self._build_limited_plan(
                    plan_factory, device_params, local_kernel_limit, output, input_, inverse)
            except LocalKernelFail:
            # One of LocalFFTKernels was out of resources.
            # Reduce the limit and try to create operations from scratch again.
                local_kernel_limit //= 2
                continue
            except GlobalKernelFail:
                raise ValueError(
                    "Could not find suitable call parameters for one of the global kernels")

            return plan

        raise ValueError("Could not find suitable call parameters for one of the local kernels")
