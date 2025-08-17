from collections.abc import Callable, Iterable

import numpy
from grunnur import (
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
from .predicates import Predicate
from .transpose import Transpose

TEMPLATE = Template.from_associated_file(__file__)


class Reduce(Computation):
    """
    Reduces the array over given axes using given binary operation.
    The resulting array shape will be set to ``1`` in the reduced axes
    (analogous to ``keepdims=True`` in ``numpy``).

    :param arr_t: an array-like defining the initial array.
    :param predicate: a :py:class:`~reikna.algorithms.Predicate` object.
    :param axes: a list of non-repeating axes to reduce over.
        If ``None``, array will be reduced over all axes.
    :param output_arr_t: an output array metadata (the `shape` must still
        correspond to the result of reducing the original array over given axes,
        but `offset` and `strides` can be set to the desired ones).

    .. py:method:: compiled_signature(output:o, input:i)

        :param input: an array with the attributes of ``arr_t``.
        :param output: an array with the attributes of ``arr_t``,
            with its shape missing axes from ``axes``.
    """

    def __init__(
        self,
        arr_t: AsArrayMetadata,
        predicate: Predicate,
        axes: Iterable[int] | None = None,
        output_arr_t: AsArrayMetadata | None = None,
    ):
        input_ = arr_t.as_array_metadata()

        dims = len(input_.shape)

        axes = tuple(range(dims) if axes is None else sorted(helpers.wrap_in_tuple(axes)))

        if len(set(axes)) != len(axes):
            raise ValueError("Cannot reduce twice over the same axis")

        if min(axes) < 0 or max(axes) >= dims:
            raise ValueError("Axes numbers are out of bounds")

        if hasattr(predicate.empty, "dtype"):
            if input_.dtype != predicate.empty.dtype:
                raise ValueError("The predicate and the array must use the same data type")
            empty = predicate.empty
        else:
            empty = numpy.asarray(predicate.empty, dtype=input_.dtype)

        remaining_axes = tuple(a for a in range(dims) if a not in axes)
        output_shape = tuple(1 if a in axes else input_.shape[a] for a in range(dims))

        # If the reduction axes are not the inner ones,
        # we will need to transpose before reducing to make them inner.
        if axes == tuple(range(dims - len(axes), dims)):
            self._transpose_axes = None
        else:
            self._transpose_axes = remaining_axes + axes

        self._reduce_dims = len(axes)
        self._operation = predicate.operation
        self._empty = empty

        output = (
            output_arr_t.as_array_metadata()
            if output_arr_t is not None
            else ArrayMetadata(shape=output_shape, dtype=input_.dtype)
        )

        if output.dtype != input_.dtype:
            raise ValueError(
                "The dtype of the output array must be the same as that of the input array"
            )
        if output.shape != output_shape:
            raise ValueError(
                "Expected the output array shape "
                + str(output_shape)
                + ", got "
                + str(output.shape)
            )

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(output, "o")),
                Parameter("input", Annotation(input_, "i")),
            ],
        )

    def _build_plan_for_wg_size(
        self,
        plan_factory: Callable[[], ComputationPlan],
        warp_size: int,
        max_wg_size: int,
        output: KernelArgument,
        input_: KernelArgument,
    ) -> ComputationPlan:
        plan = plan_factory()

        # Using algorithm cascading: sequential reduction, and then the parallel one.
        # According to Brent's theorem, the optimal sequential size is O(log(n)).
        # Setting it to the nearest power of 2 to simplify integer operations.
        max_seq_size = helpers.bounding_power_of_2(helpers.log2(max_wg_size))
        max_reduce_power = max_wg_size * max_seq_size

        if self._transpose_axes is None:
            # normal reduction
            cur_input = input_
        else:
            transpose = Transpose(input_, axes=self._transpose_axes)
            tr_output = plan.temp_array_like(transpose.parameter.output)
            plan.computation_call(transpose, tr_output, input_)

            cur_input = tr_output

        # At this point we need to reduce the innermost `_reduce_dims` axes

        axis_start = len(input_.shape) - self._reduce_dims
        axis_end = len(input_.shape) - 1

        input_slices = (axis_start, axis_end - axis_start + 1)

        part_size = helpers.product(cur_input.shape[axis_start:])
        final_size = helpers.product(cur_input.shape[:axis_start])

        while part_size > 1:
            if part_size > max_reduce_power:
                seq_size = max_seq_size
                block_size = max_wg_size
                blocks_per_part = helpers.min_blocks(part_size, block_size * seq_size)
                cur_output = plan.temp_array((final_size, blocks_per_part), input_.dtype)
                output_slices = (1, 1)
            else:
                if part_size > max_wg_size:
                    seq_size = helpers.min_blocks(part_size, max_wg_size)
                    block_size = max_wg_size
                else:
                    seq_size = 1
                    block_size = helpers.bounding_power_of_2(part_size)
                blocks_per_part = 1
                cur_output = output
                output_slices = (len(cur_output.shape), 0)

            if part_size % (block_size * seq_size) != 0:
                last_block_size = part_size % (block_size * seq_size)
            else:
                last_block_size = block_size * seq_size

            render_kwds = dict(
                dtypes=dtypes,
                seq_size=seq_size,
                blocks_per_part=blocks_per_part,
                last_block_size=last_block_size,
                log2=helpers.log2,
                block_size=block_size,
                warp_size=warp_size,
                empty=self._empty,
                operation=self._operation,
                input_slices=input_slices,
                output_slices=output_slices,
            )

            plan.kernel_call(
                TEMPLATE.get_def("reduce"),
                [cur_output, cur_input],
                kernel_name="kernel_reduce",
                global_size=(final_size, blocks_per_part * block_size),
                local_size=(1, block_size),
                render_kwds=render_kwds,
            )

            part_size = blocks_per_part
            cur_input = cur_output
            input_slices = output_slices

        return plan

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        max_wg_size = device_params.max_total_local_size

        output = args.output
        input_ = args.input

        while max_wg_size >= 1:
            try:
                plan = self._build_plan_for_wg_size(
                    plan_factory, device_params.warp_size, max_wg_size, output, input_
                )
            except VirtualSizeError:
                max_wg_size //= 2
                continue

            return plan

        raise ValueError("Could not find suitable call parameters for one of the local kernels")
