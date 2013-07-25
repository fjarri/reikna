from reikna.cluda import Snippet
import reikna.helpers as helpers
from reikna.cluda import dtypes
from reikna.core import Computation, Parameter, Annotation, Type
from reikna.transpose import Transpose

TEMPLATE = helpers.template_for(__file__)


class Predicate:

    def __init__(self, operation, empty):
        self.operation = operation
        self.empty = empty

    def __process_modules__(self, process):
        return Predicate(process(self.operation), self.empty)


def predicate_sum(dtype):
    return Predicate(
        Snippet.create(lambda v1, v2: "return ${v1} + ${v2};"),
        dtypes.c_constant(dtypes.cast(dtype)(0)))


class Reduce(Computation):
    """
    Reduces the array over given axis using given binary operation.
    """

    def __init__(self, arr_t, predicate, axes=None):

        if axes is None:
            axes = list(range(len(arr_t.shape)))

        # we require sequential axes
        assert list(axes) == list(range(axes[0], axes[-1] + 1))
        output_shape = arr_t.shape[:axes[0]] + arr_t.shape[axes[-1]+1:]
        if len(output_shape) == 0:
            output_shape = (1,)

        self._axis_start = axes[0]
        self._axis_end = axes[-1]
        self._predicate = predicate

        Computation.__init__(self, [
            Parameter('output', Annotation(Type(arr_t.dtype, shape=output_shape), 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        # FIXME: may fail if the user passes particularly sophisticated operation
        max_reduce_power = device_params.max_work_group_size

        axis_start = self._axis_start
        axis_end = self._axis_end

        if axis_end == len(input_.shape) - 1:
            # normal reduction
            cur_input = input_
        else:
            initial_axes = tuple(range(len(input_.shape)))
            tr_axes = (
                initial_axes[:axis_start] +
                initial_axes[axis_end+1:] +
                initial_axes[axis_start:axis_end+1])

            transpose = Transpose(input_, axes=tr_axes)
            tr_output = plan.temp_array_like(transpose.parameter.output)
            plan.computation_call(transpose, tr_output, input_)

            cur_input = tr_output
            axis_start = len(input_.shape) - 1 - (axis_end - axis_start)
            axis_end = len(input_.shape) - 1

        input_slices = (axis_start, axis_end - axis_start + 1)

        external_shape = cur_input.shape[:axis_start]
        part_size = helpers.product(cur_input.shape[axis_start:])
        final_size = helpers.product(external_shape)

        while part_size > 1:

            if part_size >= max_reduce_power:
                block_size = max_reduce_power
                blocks_per_part = helpers.min_blocks(part_size, block_size)
                cur_output = plan.temp_array(
                    (final_size, blocks_per_part), input_.dtype)
                output_slices = (1, 1)
            else:
                block_size = helpers.bounding_power_of_2(part_size)
                blocks_per_part = 1
                cur_output = output
                output_slices = (len(cur_output.shape), 0)

            if part_size % block_size != 0:
                last_block_size = part_size % block_size
            else:
                last_block_size = block_size

            render_kwds = dict(
                blocks_per_part=blocks_per_part,
                last_block_size=last_block_size,
                log2=helpers.log2, block_size=block_size,
                warp_size=device_params.warp_size,
                predicate=self._predicate,
                input_slices=input_slices,
                output_slices=output_slices)

            plan.kernel_call(
                TEMPLATE.get_def('reduce'),
                [cur_output, cur_input],
                global_size=(blocks_per_part * block_size, final_size),
                local_size=(block_size, 1),
                render_kwds=render_kwds)

            part_size = blocks_per_part
            cur_input = cur_output
            input_slices = output_slices

        return plan
