import numpy

from reikna.helpers import *
from reikna.core import *

TEMPLATE = template_for(__file__)


def transpose_shape(shape, axes):
    return tuple(shape[i] for i in axes)

def transpose_axes(axes, b_start, c_start):
    return axes[:b_start] + axes[c_start:] + axes[b_start:c_start]

def possible_transposes(n):
    for b in range(n - 1):
        for c in range(b + 1, n):
            yield b, c

def get_operations(source, target):
    visited = set([source])
    actions = list(possible_transposes(len(source)))

    def traverse(node, breadcrumbs, current_best):
        if current_best is not None and len(breadcrumbs) >= len(current_best):
            return current_best

        for b, c in actions:
            result = transpose_axes(node, b, c)
            if result in visited and result != target:
                continue
            visited.add(result)

            new_breadcrumbs = breadcrumbs + ((b, c),)

            if result == target:
                if current_best is None or len(current_best) > len(new_breadcrumbs):
                    return new_breadcrumbs

            current_best = traverse(result, new_breadcrumbs, current_best)
        return current_best

    return traverse(source, tuple(), None)

def get_transposes(shape, axes=None):

    source = tuple(range(len(axes)))
    if axes is None:
        axes = tuple(reversed(axes))
    else:
        assert set(source) == set(axes)

    for i in range(len(source) - 1, 0, -1):
        if source[:i] == axes[:i]:
            result = get_operations(source[i:], axes[i:])
            prefix = source[:i]
            break
    else:
        result = get_operations(source, axes)
        prefix = tuple()

    operations = [(b + len(prefix), c + len(prefix)) for b, c in result]

    transposes = []
    for b, c in operations:
        transposes.append((shape[:b], shape[b:c], shape[c:]))
        shape = transpose_axes(shape, b, c)
    return transposes



class Transpose(Computation):
    """
    Changes the order of axes in a multidimensional array.
    Works analogous to ``numpy.transpose``.

    .. py:method:: prepare_for(output, input, axes=None)

        :param output: output array
        :param input: input array
        :param axes: tuple with the new axes order.
            If ``None``, then axes will be reversed.
    """

    def __init__(self, arr, axes=None, block_width_override=None):

        self._block_width_override = block_width_override

        all_axes = range(len(arr.shape))
        if axes is None:
            axes = tuple(reversed(all_axes))
        else:
            assert set(axes) == set(all_axes)

        self._axes = tuple(axes)

        output_shape = transpose_shape(arr.shape, self._axes)
        output_arr = Type(arr.dtype, output_shape)

        Computation.__init__(self, [
            Parameter('output', Annotation(output_arr, 'o')),
            Parameter('input', Annotation(arr, 'i'))])

    def _add_transpose(self, plan, device_params,
            output_name, input_name, batch_shape, height_shape, width_shape):

        bso = self._block_width_override
        block_width = device_params.local_mem_banks if bso is None else bso

        if block_width ** 2 > device_params.max_work_group_size:
            # If it is not CPU, current solution may affect performance
            block_width = int(numpy.sqrt(device_params.max_work_group_size))

        input_height = product(height_shape)
        input_width = product(width_shape)
        batch = product(batch_shape)

        blocks_per_matrix = min_blocks(input_height, block_width)
        grid_width = min_blocks(input_width, block_width)

        render_kwds = dict(
            input_width=input_width, input_height=input_height, batch=batch,
            block_width=block_width,
            grid_width=grid_width,
            blocks_per_matrix=blocks_per_matrix,
            input_slices=[len(batch_shape), len(height_shape), len(width_shape)],
            output_slices=[len(batch_shape), len(width_shape), len(height_shape)])

        plan.kernel_call(
            TEMPLATE.get_def('transpose'), [output_name, input_name],
            global_size=(grid_width * block_width, blocks_per_matrix * block_width, batch),
            local_size=(block_width, block_width, 1),
            render_kwds=render_kwds)

    def _build_plan(self, plan_factory, device_params, output, input):
        plan = plan_factory()
        transposes = get_transposes(input.shape, self._axes)

        for i, tr in enumerate(transposes):

            batch_shape, height_shape, width_shape = tr

            mem_in = input if i == 0 else mem_out
            if i == len(transposes) - 1:
                mem_out = output
            else:
                mem_out = plan.temp_array(
                    batch_shape + width_shape + height_shape, output.dtype)

            self._add_transpose(plan, device_params,
                mem_out, mem_in, batch_shape, height_shape, width_shape)

        return plan
