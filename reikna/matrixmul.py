import numpy

from reikna.helpers import *
from reikna.core import *
import reikna.cluda.dtypes as dtypes
from reikna.cluda import OutOfResourcesError
from reikna.cluda import functions

TEMPLATE = template_for(__file__)


class MatrixMul(Computation):
    """
    Multiplies two matrices using last two dimensions and batching over remaining dimensions.
    For batching to work, the products of remaining dimensions should be equal
    (then the multiplication will be performed piecewise), or one of them should equal 1
    (then the multiplication will be batched over the remaining dimensions of the other matrix).
    """

    def __init__(self, a_arr, b_arr, out_arr=None, block_width_override=None):

        if len(a_arr.shape) == 1:
            a_arr = Type(a_arr.dtype, shape=(1,) + a_arr.shape)

        if len(b_arr.shape) == 1:
            b_arr = Type(b_arr.dtype, shape=b_arr.shape + (1,))

        if out_arr is None:
            out_dtype = dtypes.result_type(a_arr.dtype, b_arr.dtype)

            a_batch = product(a_arr.shape[:-2])
            out_shape = (
                (b_arr.shape[:-2] if a_batch == 1 else a_arr.shape[:-2]) +
                (a_arr.shape[-2], b_arr.shape[-1]))

            out_arr = Type(out_dtype, shape=out_shape)

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('a', Annotation(a_arr, 'i')),
            Parameter('b', Annotation(b_arr, 'i'))])

        self._block_width_override = block_width_override

    def _build_plan(self, plan_factory, device_params):
        bwo = self._block_width_override

        if bwo is not None:
            block_widths = [bwo]
        else:
            nbanks = device_params.local_mem_banks
            block_widths = [2 ** n for n in xrange(log2(nbanks), -1, -1)]

        a_height = self.a.shape[-2]
        a_width = self.a.shape[-1]
        b_width = self.b.shape[-1]
        a_batch = product(self.a.shape[:-2])
        b_batch = product(self.b.shape[:-2])
        batch = max(a_batch, b_batch)

        for block_width in block_widths:

            plan = plan_factory()

            if block_width ** 2 > device_params.max_work_group_size:
                continue

            num_steps = min_blocks(a_width, block_width)
            blocks_per_matrix = min_blocks(a_height, block_width)
            grid_width = min_blocks(b_width, block_width)

            render_kwds = dict(
                batched_a=(a_batch != 1),
                batched_b=(b_batch != 1),
                a_height=a_height,
                b_width=b_width,
                a_width=a_width,
                num_steps=num_steps,
                a_slices=(len(self.a.shape) - 2, 1, 1),
                b_slices=(len(self.b.shape) - 2, 1, 1),
                output_slices=(len(self.output.shape) - 2, 1, 1),
                block_width=block_width,
                mul=functions.mul(self.a.dtype, self.b.dtype, out_dtype=self.output.dtype))

            try:
                plan.kernel_call(
                    TEMPLATE.get_def('matrixmul'),
                    [self.output, self.a, self.b],
                    global_size=(
                        grid_width * block_width,
                        blocks_per_matrix * block_width,
                        batch),
                    local_size=(block_width, block_width, 1),
                    render_kwds=render_kwds,
                    dependencies=[(self.output, self.a), (self.output, self.b)])
            except OutOfResourcesError:
                continue

            return plan

        raise ValueError("Could not find suitable call parameters for the kernel")
