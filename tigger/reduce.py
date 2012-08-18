import numpy
from tigger.cluda import dtypes
from tigger.core import *
from tigger.transpose import Transpose

TEMPLATE = template_for(__file__)


def reduced_shape(shape, axis):
    l = list(shape)
    l.pop(axis)
    return tuple(l)


class Reduce(Computation):

    def _get_default_basis(self):
        return dict(shape=(1,1), dtype=numpy.float32, axis=-1,
            operation="return val1 + val2;")

    def _construct_basis(self, output, input, axis=None, operation=None):
        assert input.dtype == output.dtype
        assert input.size % output.size == 0
        assert input.size > output.size

        bs = dict(shape=input.shape, dtype=input.dtype)

        if axis is not None:
            bs['axis'] = axis
        if operation is not None:
            bs['operation'] = operation

        return bs

    def _get_base_signature(self):
        bs = self._basis
        return ([('output', ArrayValue(reduced_shape(bs.shape, bs.axis), bs.dtype))],
            [('input', ArrayValue(bs.shape, bs.dtype))],
            [])

    def _construct_kernels(self):
        bs = self._basis
        operations = []

        # may fail if the user passes particularly sophisticated operation
        max_reduce_power = self._ctx.device_params.max_block_size

        axis = bs.axis if bs.axis >= 0 else len(bs.shape) + axis

        size = product(bs.shape)
        final_size = product(reduced_shape(bs.shape, bs.axis))

        if len(bs.shape) == 1 or axis == len(bs.shape) - 1:
            # normal reduction
            input_name = 'input'
        elif axis == 0:
            transpose = Transpose(self._ctx)
            tr_shape = (bs.shape[0], product(bs.shape[1:]))
            transpose.prepare(dtype=dtypes.normalize_type(bs.dtype),
                input_height=tr_shape[0],
                input_width=tr_shape[1])
            operations.append(Allocate('_tr_output', (tr_shape[1], tr_shape[0]), bs.dtype))
            operations.append(ComputationCall(
                transpose, '_tr_output', 'input'))
            input_name = '_tr_output'
        else:
            raise NotImplementedError()

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
                operation_code=bs.operation)

            operations.append(KernelCall(
                'reduce', [output_name, input_name], src,
                global_size=global_size, local_size=block_size
            ))

            size = new_size
            input_name = output_name

        return operations

