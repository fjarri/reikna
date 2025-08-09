from typing import Callable

from grunnur import (
    ArrayMetadata,
    AsArrayMetadata,
    DeviceParameters,
    Template,
    VirtualSizeError,
    dtypes,
    functions,
)

from .. import helpers
from ..core import Annotation, Computation, ComputationPlan, KernelArguments, Parameter, Type

TEMPLATE = Template.from_associated_file(__file__)


class MatrixMul(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    Multiplies two matrices using last two dimensions and batching over remaining dimensions.
    For batching to work, the products of remaining dimensions should be equal
    (then the multiplication will be performed piecewise), or one of them should equal 1
    (then the multiplication will be batched over the remaining dimensions of the other matrix).

    :param a_arr: an array-like defining the first argument.
    :param b_arr: an array-like defining the second argument.
    :param out_arr: an array-like definign the output; if not given, both shape and dtype
        will be derived from ``a_arr`` and ``b_arr``.
    :param block_width_override: if provided, it will used as a block size of
        the multiplication kernel.
    :param transposed_a: if ``True``, the first matrix will be transposed
        before the multiplication.
    :param transposed_b: if ``True``, the second matrix will be transposed
        before the multiplication.

    .. py:method:: compiled_signature(output:o, matrix_a:i, matrix_b:i)

        :param output: the output of matrix multiplication.
        :param matrix_a: the first argument.
        :param matrix_b: the second argument.
    """

    def __init__(
        self,
        a_arr_t: AsArrayMetadata,
        b_arr_t: AsArrayMetadata,
        out_arr_t: AsArrayMetadata | None = None,
        block_width_override: int | None = None,
        transposed_a: bool = False,
        transposed_b: bool = False,
    ):
        a_arr = a_arr_t.as_array_metadata()
        b_arr = b_arr_t.as_array_metadata()

        if len(a_arr.shape) == 1:
            a_arr = ArrayMetadata(dtype=a_arr.dtype, shape=(1,) + a_arr.shape)

        if len(b_arr.shape) == 1:
            b_arr = ArrayMetadata(dtype=b_arr.dtype, shape=b_arr.shape + (1,))

        a_batch_shape = a_arr.shape[:-2]
        b_batch_shape = b_arr.shape[:-2]
        a_outer_size = a_arr.shape[-1 if transposed_a else -2]
        convolution_size = a_arr.shape[-2 if transposed_a else -1]
        b_outer_size = b_arr.shape[-2 if transposed_b else -1]

        if out_arr_t is None:
            out_dtype = dtypes.result_type(a_arr.dtype, b_arr.dtype)

            batch_len = max(len(a_batch_shape), len(b_batch_shape))
            batch_shape = b_batch_shape if helpers.product(a_batch_shape) == 1 else a_batch_shape
            batch_shape = (1,) * (batch_len - len(batch_shape)) + batch_shape

            out_shape = batch_shape + (a_outer_size, b_outer_size)

            out_arr = ArrayMetadata(dtype=out_dtype, shape=out_shape)
        else:
            out_arr = out_arr_t.as_array_metadata()

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(out_arr, "o")),
                Parameter("matrix_a", Annotation(a_arr, "i")),
                Parameter("matrix_b", Annotation(b_arr, "i")),
            ],
        )

        self._block_width_override = block_width_override
        self._a_outer_size = a_outer_size
        self._convolution_size = convolution_size
        self._b_outer_size = b_outer_size
        self._transposed_a = transposed_a
        self._transposed_b = transposed_b

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        bwo = self._block_width_override

        output = args.output
        matrix_a = args.matrix_a
        matrix_b = args.matrix_b

        if bwo is not None:
            block_widths = [bwo]
        else:
            nbanks = device_params.local_mem_banks
            block_widths = [2**n for n in range(helpers.log2(nbanks), -1, -1)]

        a_batch = helpers.product(matrix_a.shape[:-2])
        b_batch = helpers.product(matrix_b.shape[:-2])
        batch = max(a_batch, b_batch)

        for block_width in block_widths:
            plan = plan_factory()

            if block_width**2 > device_params.max_total_local_size:
                continue

            num_steps = helpers.min_blocks(self._convolution_size, block_width)
            a_blocks = helpers.min_blocks(self._a_outer_size, block_width)
            b_blocks = helpers.min_blocks(self._b_outer_size, block_width)

            render_kwds = dict(
                dtypes=dtypes,
                batched_a=(a_batch != 1),
                batched_b=(b_batch != 1),
                transposed_a=self._transposed_a,
                transposed_b=self._transposed_b,
                num_steps=num_steps,
                a_slices=(len(matrix_a.shape) - 2, 1, 1),
                b_slices=(len(matrix_b.shape) - 2, 1, 1),
                output_slices=(len(output.shape) - 2, 1, 1),
                block_width=block_width,
                mul=functions.mul(matrix_a.dtype, matrix_b.dtype, out_dtype=output.dtype),
            )

            try:
                plan.kernel_call(
                    TEMPLATE.get_def("matrixmul"),
                    [output, matrix_a, matrix_b],
                    kernel_name="kernel_matrixmul",
                    global_size=(batch, a_blocks * block_width, b_blocks * block_width),
                    local_size=(1, block_width, block_width),
                    render_kwds=render_kwds,
                )
            except VirtualSizeError:
                continue

            return plan

        raise ValueError("Could not find suitable call parameters for the kernel")
