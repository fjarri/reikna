import numpy
import helpers
from .helpers import *


TEMPLATE = loadTemplate(__file__)


class MatrixMul(Computation):

    def _get_default_basis(self):
        return dict(a_dtype=numpy.float32, b_dtype=numpy.float32, out_dtype=numpy.float32,
            a_height=1, a_width=1, b_width=1, batch=1,
            batched_a=False, batched_b=False,
            block_size_override=None)

    def _construct_basis(self, a, b, out=None):

        bs = Computation._construct_basis(self)

        # Derive types

        if out is not None:
            bs.out_dtype = out.dtype
        else:
            bs.out_dtype = numpy.result_type(a.dtype, b.dtype)

        if self._debug:
            for dtype in (a.dtype, b.dtype, bs.out_dtype):
                assert self._env.supportsDtype(dtype)

        bs.a_dtype = a.dtype
        bs.b_dtype = b.dtype

        # Derive shapes

        if self._debug:
            assert len(a.shape) >= 2
            assert len(b.shape) >= 2
            assert a.shape[-1] == b.shape[-2]

        a_batch = product(a.shape[:-2])
        b_batch = product(b.shape[:-2])
        out_batch = max(a_batch, b_batch)

        if out is None or self._debug:
            out_shape = (b.shape[:-2] if a_batch == 1 else a.shape[:-2]) + (a.shape[-2], b.shape[-1])
        else:
            out_shape = out.shape

        if self._debug:
            assert a_batch == 1 or b_batch == 1 or a_batch == b_batch
            assert a_batch != b_batch or a.shape[:-2] == b.shape[:-2]
            if out is not None:
                assert len(out_shape) >= 2
                assert out_batch == product(out_shape[:-2])

        bs.update(dict(
            a_width=numpy.int32(a.shape[-1]),
            a_height=numpy.int32(a.shape[-2]),
            b_width=numpy.int32(b.shape[-1]),
            batch=numpy.int32(out_batch),
            batched_a=numpy.int32(a_batch == 1),
            batched_b=numpy.int32(b_batch == 1),
            out_shape=out_shape))

        return bs

    def _construct_derived(self):
        dp = Computation._construct_derived(self)
        bp = self._basis

        needs_double = is_double(bp.out_dtype)

        bso = bp.block_size_override
        block_size = self._env.params.smem_banks if bso is None else bso

        dp.blocks_per_matrix = numpy.int32(min_blocks(bp.a_height, block_size))
        dp.block_size = block_size
        dp.module = self._env.compile(TEMPLATE, bp=bp, dp=dp, helpers=helpers)
        dp.kernel_matrixmul = dp.module.matrixmul

        dp.grid = (
            int(min_blocks(bp.b_width, block_size)),
            int(dp.blocks_per_matrix * bp.batch)
        )
        dp.block = (block_size, block_size, 1)
        dp.shared = block_size * block_size * (bp.a_dtype.itemsize + bp.b_dtype.itemsize)

        # TODO: do we need preparation?
        #dp.kernel_matrixmul.prepare(block=block, grid=grid, enqueue=self._queue, shared=shared)

        return dp

    def _call(self, a, b, out=None):
        bp = self._basis
        dp = self._derived
        out_buf = self._env.allocate(bp.out_shape, bp.out_dtype) if out is None else out

        dp.kernel_matrixmul(out_buf, a, b, block=dp.block, grid=dp.grid)

        if out is None:
            return out_buf


class MockMatrixMul(MatrixMul):

    def _dot(self, out, a, b):
        basis = db if self._dynamic else self._basis

        ha, wa, wb, batch, batched_a, batched_b, scale = \
            basis.a_height, basis.a_width, basis.b_width, \
            basis.batch, basis.batched_a, basis.batched_b, self._scale

        a_view = a.reshape(batch if batched_a else 1, ha, wa)
        b_view = b.reshape(batch if batched_b else 1, wa, wb)
        out_view = out.reshape(batch, ha, wb)

        for i in xrange(batch):
            a_part = a_view[i] if batched_a else a_view[0]
            b_part = b_view[i] if batched_b else b_view[0]
            out_view[i] = numpy.dot(a_part, b_part)

        out_view *= scale
