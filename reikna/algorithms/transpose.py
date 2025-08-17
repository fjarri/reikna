from collections.abc import Callable, Iterable, Iterator

import numpy
from grunnur import (
    Array,
    ArrayMetadata,
    AsArrayMetadata,
    DeviceParameters,
    Template,
    VirtualSizeError,
    dtypes,
)

from .. import helpers
from ..core import (
    Annotation,
    Computation,
    ComputationPlan,
    KernelArgument,
    KernelArguments,
    Parameter,
)

TEMPLATE = Template.from_associated_file(__file__)


def transpose_shape(shape: tuple[int, ...], axes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(shape[i] for i in axes)


def transpose_axes(axes: tuple[int, ...], b_start: int, c_start: int) -> tuple[int, ...]:
    return axes[:b_start] + axes[c_start:] + axes[b_start:c_start]


def possible_transposes(shape_len: int) -> Iterator[tuple[int, int]]:
    for b_start in range(shape_len - 1):
        for c_start in range(b_start + 1, shape_len):
            yield b_start, c_start


def get_operations(source: tuple[int, ...], target: tuple[int, ...]) -> list[tuple[int, int]]:
    visited = {source}
    actions = list(possible_transposes(len(source)))

    def traverse(
        node: tuple[int, ...],
        breadcrumbs: list[tuple[int, int]],
        current_best: list[tuple[int, int]] | None,
    ) -> list[tuple[int, int]]:
        if current_best is not None and len(breadcrumbs) >= len(current_best):
            return current_best

        for b_start, c_start in actions:
            result = transpose_axes(node, b_start, c_start)
            if result in visited and result != target:
                continue
            visited.add(result)

            new_breadcrumbs = [*breadcrumbs, (b_start, c_start)]

            if result == target and (
                current_best is None or len(current_best) > len(new_breadcrumbs)
            ):
                return new_breadcrumbs

            current_best = traverse(result, new_breadcrumbs, current_best)

        # `current_best` will not be `None` as long as `actions` is not empty
        # The assertion is to appease `mypy`.
        assert current_best is not None  # noqa: S101

        return current_best

    return traverse(source, [], None)


def _get_transposes(
    shape: tuple[int, ...], axes: tuple[int, ...]
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    source = tuple(range(len(axes)))

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
    for b_start, c_start in operations:
        transposes.append((shape[:b_start], shape[b_start:c_start], shape[c_start:]))
        shape = transpose_axes(shape, b_start, c_start)
    return transposes


class Transpose(Computation):
    """
    Changes the order of axes in a multidimensional array.
    Works analogous to ``numpy.transpose``.

    :param arr_t: an array-like defining the initial array.
    :param output_arr_t: an array-like defining the output array.
        If ``None``, its shape will be derived based on the shape of ``arr_t``,
        its dtype will be equal to that of ``arr_t``,
        and any non-default offset or strides of ``arr_t`` will be ignored.
    :param axes: tuple with the new axes order.
        If ``None``, then axes will be reversed.

    .. py:function:: compiled_signature(output:o, input:i)

        :param output: an array with all the attributes of ``arr_t``,
            with the shape permuted according to ``axes``.
        :param input: an array with all the attributes of ``arr_t``.
    """

    def __init__(
        self,
        arr_t: AsArrayMetadata,
        output_arr_t: AsArrayMetadata | None = None,
        axes: Iterable[int] | None = None,
        block_width_override: int | None = None,
    ):
        self._block_width_override = block_width_override

        input_ = arr_t.as_array_metadata()

        all_axes = range(len(input_.shape))
        if axes is None:
            axes = tuple(reversed(all_axes))
        else:
            axes = tuple(axes)
            if set(axes) != set(all_axes):
                raise ValueError("All transpose axes must be distinct")

        self._axes = axes
        self._transposes = _get_transposes(input_.shape, self._axes)

        output_shape = transpose_shape(input_.shape, self._axes)
        output = (
            output_arr_t.as_array_metadata()
            if output_arr_t is not None
            else ArrayMetadata(shape=output_shape, dtype=input_.dtype)
        )

        if output.shape != output_shape:
            raise ValueError(f"Expected output array shape: {output_shape}, got {output.shape}")
        if output.dtype != input_.dtype:
            raise ValueError("Input and output array must have the same dtype")

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(output, "o")),
                Parameter("input", Annotation(input_, "i")),
            ],
        )

    def _add_transpose(
        self,
        plan: ComputationPlan,
        mem_out: KernelArgument,
        mem_in: KernelArgument,
        batch_shape: tuple[int, ...],
        height_shape: tuple[int, ...],
        width_shape: tuple[int, ...],
        block_width: int,
    ) -> None:
        input_height = helpers.product(height_shape)
        input_width = helpers.product(width_shape)
        batch = helpers.product(batch_shape)

        blocks_per_matrix = helpers.min_blocks(input_height, block_width)
        grid_width = helpers.min_blocks(input_width, block_width)

        render_kwds = dict(
            dtypes=dtypes,
            input_width=input_width,
            input_height=input_height,
            batch=batch,
            block_width=block_width,
            grid_width=grid_width,
            blocks_per_matrix=blocks_per_matrix,
            input_slices=(len(batch_shape), len(height_shape), len(width_shape)),
            output_slices=(len(batch_shape), len(width_shape), len(height_shape)),
        )

        plan.kernel_call(
            TEMPLATE.get_def("transpose"),
            [mem_out, mem_in],
            kernel_name="kernel_transpose",
            global_size=(batch, blocks_per_matrix * block_width, grid_width * block_width),
            local_size=(1, block_width, block_width),
            render_kwds=render_kwds,
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        output = args.output
        input_ = args.input

        mem_out = output
        for i, transpose in enumerate(self._transposes):
            batch_shape, height_shape, width_shape = transpose

            mem_in = input_ if i == 0 else mem_out
            if i == len(self._transposes) - 1:
                mem_out = output
            else:
                mem_out = plan.temp_array(batch_shape + width_shape + height_shape, output.dtype)

            bso = self._block_width_override
            block_width = device_params.local_mem_banks if bso is None else bso

            if block_width**2 > device_params.max_total_local_size:
                # If it is not CPU, current solution may affect performance
                block_width = int(numpy.sqrt(device_params.max_total_local_size))

            while block_width >= 1:
                try:
                    self._add_transpose(
                        plan,
                        mem_out,
                        mem_in,
                        batch_shape,
                        height_shape,
                        width_shape,
                        block_width,
                    )
                except VirtualSizeError:
                    block_width //= 2
                    continue
                break

        return plan
