import numpy
from tigger.core import *

TEMPLATE = template_for(__file__)


class MatrixMul(Computation):

    def _get_default_basis(self):
        return dict(a_dtype=numpy.float32, b_dtype=numpy.float32, out_dtype=numpy.float32,
            a_height=1, a_width=1, b_width=1, batch=1,
            batched_a=False, batched_b=False,
            block_size_override=None,
            out_shape=(1, 1))

    def _construct_basis(self, out, a, b):

        bs = AttrDict()

        if out.dtype is None:
            bs.out_dtype = dtypes.result_type(a.dtype, b.dtype)
        else:
            bs.out_dtype = out.dtype

        bs.a_dtype = a.dtype
        bs.b_dtype = b.dtype

        if self._debug:
            assert len(a.shape) >= 2
            assert len(b.shape) >= 2
            assert a.shape[-1] == b.shape[-2]

        a_batch = product(a.shape[:-2])
        b_batch = product(b.shape[:-2])
        out_batch = max(a_batch, b_batch)

        if out.shape is None:
            out_shape = (b.shape[:-2] if a_batch == 1 else a.shape[:-2],
                a.shape[-2], b.shape[-1])
        else:
            out_shape = out.shape

        if self._debug:
            assert a_batch == 1 or b_batch == 1 or a_batch == b_batch
            assert a_batch != b_batch or a.shape[:-2] == b.shape[:-2]
            if out.shape is not None:
                assert len(out_shape) >= 2
                assert out_batch == product(out_shape[:-2])

        bs.update(dict(
            a_width=a.shape[-1],
            a_height=a.shape[-2],
            b_width=b.shape[-1],
            batch=out_batch,
            batched_a=(a_batch == 1),
            batched_b=(b_batch == 1),
            out_shape=out_shape))

        return bs

    def _get_base_signature(self, basis):

        a_shape = (basis.batch if basis.batched_a else 1, basis.a_height, basis.a_width)
        b_shape = (basis.batch if basis.batched_b else 1, basis.a_width, basis.b_width)

        return (
            [('out', ArrayValue(basis.out_shape, basis.out_dtype))],
            [
                ('a', ArrayValue(a_shape, basis.a_dtype)),
                ('b', ArrayValue(b_shape, basis.b_dtype))
            ],
            [])

    def _construct_operations(self, basis, operations):

        bso = basis.block_size_override
        block_width = self._ctx.device_params.smem_banks if bso is None else bso
        block_size = block_width ** 2

        if block_size > self._ctx.device_params.max_block_size:
            # If it is not CPU, current solution may affect performance
            block_width = int(numpy.sqrt(self._ctx.device_params.max_block_size))
            block_size = block_width ** 2

        blocks_per_matrix = min_blocks(basis.a_height, block_width)
        grid_width = min_blocks(basis.b_width, block_width)
        grid_size = grid_width * blocks_per_matrix * basis.batch

        shared = block_size * (basis.a_dtype.itemsize + basis.b_dtype.itemsize)

        render_kwds = dict(
            block_width=block_width,
            grid_width=grid_width,
            blocks_per_matrix=blocks_per_matrix)

        operations.add_kernel(
            TEMPLATE, 'matrixmul',
            ['out', 'a', 'b'],
            global_size=grid_size * block_size, local_size=block_size, shared=shared,
            render_kwds=render_kwds
        )
