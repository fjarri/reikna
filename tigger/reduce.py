import numpy
from tigger.core import *

TEMPLATE = template_for(__file__)


class Reduce(Computation):

    def __init__(self, ctx, code, **kwds):
        Computation.__init__(self, ctx, **kwds)
        self._code = code

    def _get_default_basis(self):
        return dict(input_size=1, output_size=1, batch=1, dtype=numpy.float32)

    def _construct_basis(self, output, input):
        assert input.dtype == output.dtype

        input_size = input.shape[-1]
        output_size = output.shape[-1]
        input_batch = product(input.shape[:-1])
        output_batch = product(output.shape[:-1])

        assert input_size % output_size == 0
        assert input_size > output_size
        assert input_batch == output_batch

        return dict(input_size=input_size, output_size=output_size,
            dtype=input.dtype, batch=input_batch)

    def _get_base_signature(self):
        bs = self._basis
        input_shape = (bs.batch, bs.input_size) if bs.batch > 1 else (bs.input_size,)
        output_shape = (bs.batch, bs.output_size) if bs.batch > 1 else (bs.output_size,)
        return ([('output', ArrayValue(output_shape, bs.dtype))],
            [('input', ArrayValue(input_shape, bs.dtype))],
            [])

    def _construct_kernels(self):
        bs = self._basis
        operations = []

        # may fail if the user passes particularly sophisticated operation
        max_reduce_power = self._ctx.device_params.max_block_size

        data_in = None
        size = bs.input_size * bs.batch
        final_size = bs.output_size * bs.batch

        input_name = 'input'
        reduction_stage = 0
        while size > final_size:
            reduction_stage += 1

            part_size = size / final_size

            if part_size >= max_reduce_power:
                block_size = max_reduce_power
                blocks_per_part = min_blocks(part_size, block_size)
                blocks_num = blocks_per_part * final_size
                last_block_size = part_size - (blocks_per_part - 1) * block_size
                new_size = blocks_num
            else:
                block_size = 2 ** (log2(size / final_size - 1) + 1)
                blocks_per_part = 1
                blocks_num = final_size
                last_block_size = size / final_size
                new_size = final_size

            global_size = blocks_num * block_size

            if new_size != final_size:
                temp_name = '_reduce_temp' + str(reduction_stage)
                operations.append(Allocate(temp_name, (new_size,), bs.dtype))
                output_name = temp_name
            else:
                output_name = 'output'

            src = self._render(TEMPLATE,
                input_name=input_name, output_name=output_name,
                internal_inputs=[input_name],
                internal_outputs=[output_name],
                blocks_per_part=blocks_per_part, last_block_size=last_block_size,
                log2=log2, block_size=block_size,
                warp_size=self._ctx.device_params.warp_size,
                operation_code=self._code)

            operations.append(KernelCall(
                'reduce', [output_name, input_name], src,
                global_size=global_size, local_size=block_size
            ))

            size = new_size
            input_name = output_name

        return operations

