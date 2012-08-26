import numpy
from tigger.core import *

TEMPLATE = template_for(__file__)


def transpose_shape(shape, axes):
    return tuple(shape[i] for i in axes)

def transpose(axes, b_start, c_start):
    return axes[:b_start] + axes[c_start:] + axes[b_start:c_start]

def possible_transposes(n):
    for b in xrange(n - 1):
        for c in xrange(b + 1, n):
            yield b, c

def get_operations(source, target):
    visited = set([source])
    actions = list(possible_transposes(len(source)))

    def traverse(node, breadcrumbs, current_best):
        if current_best is not None and len(breadcrumbs) >= len(current_best):
            return current_best

        for b, c in actions:
            result = transpose(node, b, c)
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

    for i in xrange(len(source) - 1, 0, -1):
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
        transposes.append((product(shape[:b]), product(shape[b:c]), product(shape[c:])))
        shape = transpose(shape, b, c)
    return transposes



class Transpose(Computation):

    def _get_default_basis(self):
        return dict(dtype=numpy.float32, input_shape=(1, 1), axes=(1, 0),
            block_size_override=None)

    def _construct_basis(self, output, input, axes=None):

        bs = AttrDict()

        assert output.dtype is None or output.dtype == input.dtype
        bs.dtype = input.dtype
        bs.input_shape = input.shape

        assert product(output.shape) == product(input.shape)

        if axes is None:
            axes = tuple(reversed(xrange(len(input.shape))))
        else:
            assert set(axes) == set(xrange(len(input.shape)))
        bs.axes = tuple(axes)

        return bs

    def _get_base_signature(self, basis):

        output_shape = transpose_shape(basis.input_shape, basis.axes)

        return (
            [('output', ArrayValue(output_shape, basis.dtype))],
            [('input', ArrayValue(basis.input_shape, basis.dtype))],
            [])

    def _add_transpose(self, basis, device_params, operations,
            output_name, input_name, batch, input_height, input_width):

        bso = basis.block_size_override
        block_width = device_params.smem_banks if bso is None else bso

        if block_width ** 2 > device_params.max_work_group_size:
            # If it is not CPU, current solution may affect performance
            block_width = int(numpy.sqrt(device_params.max_work_group_size))

        blocks_per_matrix = min_blocks(input_height, block_width)
        grid_width = min_blocks(input_width, block_width)

        shared = block_width * (block_width + 1) * basis.dtype.itemsize

        render_kwds = dict(
            input_width=input_width, input_height=input_height, batch=batch,
            block_width=block_width,
            grid_width=grid_width,
            blocks_per_matrix=blocks_per_matrix)

        operations.add_kernel(
            TEMPLATE, 'transpose', [output_name, input_name],
            global_size=(grid_width * block_width, blocks_per_matrix * batch * block_width),
            local_size=(block_width, block_width),
            shared=shared, render_kwds=render_kwds)

    def _construct_operations(self, basis, device_params, operations):
        transposes = get_transposes(basis.input_shape, basis.axes)

        temp_shape = (product(basis.input_shape),)
        if len(transposes) == 1:
            args = [('output', 'input')]
        elif len(transposes) == 2:
            operations.add_allocation('_tr_temp', temp_shape, basis.dtype)
            args = [
                ('_tr_temp', 'input'),
                ('output', '_tr_temp')
            ]
        else:
            tnames = ['_tr_temp1', '_tr_temp2']
            operations.add_allocation(tnames[0], temp_shape, basis.dtype)
            operations.add_allocation(tnames[1], temp_shape, basis.dtype)

            iname = 'input'
            oname = tnames[0]
            args = [(oname, iname)]
            other_tname = lambda name: tnames[0] if name == tnames[1] else tnames[1]
            for i in xrange(1, len(transposes)):
                iname = oname
                oname = 'output' if i == len(transposes) - 1 else other_tname(iname)
                args.append((oname, iname))

        for tr, arg_pair in zip(transposes, args):
            batch, height, width = tr
            oname, iname = arg_pair
            self._add_transpose(basis, device_params, operations,
                oname, iname, batch, height, width)
