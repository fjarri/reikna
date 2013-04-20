import numpy

from reikna.helpers import *
from reikna.core import *
from reikna.cluda import OutOfResourcesError
from reikna.cluda import functions

TEMPLATE = template_for(__file__)


class MatrixMul(Computation):
    """
    Multiplies two matrices using last two dimensions and batching over remaining dimensions.
    For batching to work, the products of remaining dimensions should be equal
    (then the multiplication will be performed piecewise), or one of them should equal 1
    (then the multiplication will be batched over the remaining dimensions of the other matrix).

    .. py:method:: prepare_for(out, a, b)

        :param out: buffer for the result
        :param a: first matrix (if it is a vector, it will be reshaped to a matrix with one row)
        :param b: second matrix
    """

    def _get_argnames(self):
        return ('out',), ('a', 'b'), tuple()

    def _get_basis_for(self, out, a, b, block_width_override=None):

        bs = AttrDict(block_width_override=block_width_override)

        if out.dtype is None:
            bs.out_dtype = dtypes.result_type(a.dtype, b.dtype)
        else:
            bs.out_dtype = out.dtype

        bs.a_dtype = a.dtype
        bs.b_dtype = b.dtype

        a_shape = a.shape if len(a.shape) > 1 else (1, a.shape[0])
        b_shape = b.shape

        if self._debug:
            assert len(b_shape) >= 2
            assert a_shape[-1] == b_shape[-2]

        a_batch = product(a_shape[:-2])
        b_batch = product(b_shape[:-2])
        out_batch = max(a_batch, b_batch)

        if out.shape is None:
            out_shape = (b_shape[:-2] if a_batch == 1 else a_shape[:-2],
                a_shape[-2], b_shape[-1])
        else:
            out_shape = out.shape

        if self._debug:
            assert a_batch == 1 or b_batch == 1 or a_batch == b_batch
            assert a_batch != b_batch or a_shape[:-2] == b_shape[:-2]
            if out.shape is not None:
                assert len(out_shape) >= 2
                assert out_batch == product(out_shape[:-2])

        bs.update(dict(
            a_width=a_shape[-1],
            a_height=a_shape[-2],
            b_width=b_shape[-1],
            batch=out_batch,
            batched_a=(a_batch != 1),
            batched_b=(b_batch != 1),
            out_shape=out_shape))

        return bs

    def _get_argvalues(self, basis):

        a_shape = (basis.batch if basis.batched_a else 1, basis.a_height, basis.a_width)
        b_shape = (basis.batch if basis.batched_b else 1, basis.a_width, basis.b_width)

        return dict(
            out=ArrayValue(basis.out_shape, basis.out_dtype),
            a=ArrayValue(a_shape, basis.a_dtype),
            b=ArrayValue(b_shape, basis.b_dtype))

    def _construct_operations(self, basis, device_params):
        bwo = basis.block_width_override

        if bwo is not None:
            block_widths = [bwo]
        else:
            nbanks = device_params.local_mem_banks
            block_widths = [2 ** n for n in xrange(log2(nbanks), -1, -1)]

        for block_width in block_widths:

            operations = self._get_operation_recorder()

            if block_width ** 2 > device_params.max_work_group_size:
                continue

            blocks_per_matrix = min_blocks(basis.a_height, block_width)
            grid_width = min_blocks(basis.b_width, block_width)

            render_kwds = dict(
                block_width=block_width,
                grid_width=grid_width,
                blocks_per_matrix=blocks_per_matrix,
                mul=functions.mul(basis.a_dtype, basis.b_dtype, out_dtype=basis.out_dtype))

            try:
                operations.add_kernel(
                    TEMPLATE.get_def('matrixmul'), ['out', 'a', 'b'],
                    global_size=(grid_width * block_width,
                        blocks_per_matrix * basis.batch * block_width),
                    local_size=(block_width, block_width),
                    render_kwds=render_kwds,
                    dependencies=[('out', 'a'), ('out', 'b')])
            except OutOfResourcesError:
                continue

            return operations

        raise ValueError("Could not find suitable call parameters for the kernel")
