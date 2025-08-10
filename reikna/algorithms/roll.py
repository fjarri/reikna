from typing import Callable, Iterable

import numpy
from grunnur import (
    ArrayMetadata,
    AsArrayMetadata,
    DeviceParameters,
    Template,
)

from reikna import transformations

from ..core import (
    Annotation,
    Computation,
    ComputationPlan,
    KernelArgument,
    KernelArguments,
    Parameter,
    Type,
)
from .pureparallel import PureParallel


class Roll(Computation):
    def __init__(self, array: AsArrayMetadata, axis: int = -1):
        array = array.as_array_metadata()

        self._axis = axis % len(array.shape)
        self._comp = PureParallel(
            [
                Parameter("output", Annotation(array, "o")),
                Parameter("input", Annotation(array, "i")),
                Parameter("shift", Annotation(Type.scalar(numpy.int32))),
            ],
            """
            <%
                shape = input.shape
            %>
            %for i in range(len(shape)):
                VSIZE_T output_${idxs[i]} =
                    %if i == axis:
                    ${shift} == 0 ?
                        ${idxs[i]} :
                        // Since ``shift`` can be negative, and its absolute value greater than
                        // ``shape[i]``, a double modulo division is necessary
                        // (the ``%`` operator preserves the sign of the dividend in C).
                        (${idxs[i]} + (${shape[i]} + ${shift} % ${shape[i]})) % ${shape[i]};
                    %else:
                    ${idxs[i]};
                    %endif
            %endfor
            ${output.store_idx}(
                ${", ".join("output_" + name for name in idxs)},
                ${input.load_idx}(${", ".join(idxs)}));
            """,
            guiding_array="input",
            render_kwds=dict(axis=self._axis),
        )

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(array, "o")),
                Parameter("input", Annotation(array, "i")),
                Parameter("shift", Annotation(numpy.int32)),
            ],
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()
        plan.computation_call(self._comp, args.output, args.input, args.shift)
        return plan


class RollInplace(Computation):
    def __init__(self, array: AsArrayMetadata, axis: int = -1):
        array = array.as_array_metadata()
        self._comp = Roll(array, axis=axis)
        Computation.__init__(
            self,
            [
                Parameter("array", Annotation(array, "io")),
                Parameter("shift", Annotation(numpy.int32)),
            ],
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        array = args.array
        shift = args.shift

        temp = plan.temp_array_like(array)
        plan.computation_call(self._comp, temp, array, shift)

        tr = transformations.copy(temp, out_arr_t=array)
        copy_comp = PureParallel.from_trf(tr, guiding_array=tr.parameter.output)
        plan.computation_call(copy_comp, array, temp)

        return plan
