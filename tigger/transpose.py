import numpy
from tigger.core import *

TEMPLATE = template_for(__file__)


class Transpose(Computation):

    def _get_default_basis(self):
        return dict(dtype=numpy.float32, input_height=1, input_width=1,
            batch=1, block_size_override=None)

    def _construct_basis(self, output, input):

        bs = AttrDict()

        assert output.dtype is None or output.dtype == input.dtype
        bs.dtype = input.dtype

        bs.input_width = input.shape[-1]
        bs.input_height = input.shape[-2]
        bs.batch = product(input.shape[:-2])

        return bs

    def _get_base_signature(self):
        bs = self._basis
        input_shape = ((bs.batch,) if bs.batch > 1 else ()) + (bs.input_height, bs.input_width)
        output_shape = ((bs.batch,) if bs.batch > 1 else ()) + (bs.input_width, bs.input_height)

        return (
            [('output', ArrayValue(output_shape, bs.dtype))],
            [('input', ArrayValue(input_shape, bs.dtype))],
            [])

    def _construct_kernels(self):
        bs = self._basis

        bso = bs.block_size_override
        block_width = self._ctx.device_params.smem_banks if bso is None else bso
        block_size = block_width ** 2

        if block_size > self._ctx.device_params.max_block_size:
            # If it is not CPU, current solution may affect performance
            block_width = int(numpy.sqrt(self._ctx.device_params.max_block_size))
            block_size = block_width ** 2

        blocks_per_matrix = min_blocks(bs.input_height, block_width)
        grid_width = min_blocks(bs.input_width, block_width)
        grid_size = grid_width * blocks_per_matrix * bs.batch

        shared = block_width * (block_width + 1) * bs.dtype.itemsize

        src = self._render(TEMPLATE,
            block_width=block_width,
            grid_width=grid_width,
            blocks_per_matrix=blocks_per_matrix)

        return [KernelCall(
            'transpose', ['output', 'input'], src,
            global_size=grid_size * block_size, local_size=block_size, shared=shared
        )]
