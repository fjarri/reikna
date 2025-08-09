from typing import Callable, Iterable

import numpy
from grunnur import AsArrayMetadata, DeviceParameters, Template, dtypes

from .. import helpers
from ..algorithms import PureParallel
from ..core import Annotation, Computation, ComputationPlan, KernelArguments, Parameter
from ..transformations import copy

TEMPLATE = Template.from_associated_file(__file__)


class FFTShift(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    Shift the zero-frequency component to the center of the spectrum.
    The interface is similar to ``numpy.fft.fftshift`` (or ``ifftshift`` when ``inverse == True``),
    and the output is the same for the same array shape and axes.

    :param arr_t: an array-like defining the problem array.
    :param axes: a tuple with axes over which to perform the shift.
        If not given, the shift is performed over all the axes.

    .. py:method:: compiled_signature(output:o, input:i)

        ``output`` and ``input`` may be the same array.

        :param output: an array with the attributes of ``arr_t``.
        :param input: an array with the attributes of ``arr_t``.
        :param inverse: a scalar value castable to integer.
            If ``0``, the forward transform is applied (equivalent of ``numpy.fft.fftshift``),
            if ``1``, the inverse one (equivalent of ``numpy.fft.ifftshift``).
    """

    def __init__(self, arr_t: AsArrayMetadata, axes: Iterable[int] | None = None):
        arr = arr_t.as_array_metadata()
        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(arr, "o")),
                Parameter("input", Annotation(arr, "i")),
                Parameter("inverse", Annotation(numpy.int32), default=0),
            ],
        )

        if axes is None:
            axes = tuple(range(len(arr.shape)))
        else:
            axes = tuple(axes)
        self._axes = axes

    def _build_trivial_plan(
        self, plan_factory: Callable[[], ComputationPlan], args: KernelArguments
    ) -> ComputationPlan:
        # Trivial problem. Need to add a dummy kernel
        # because we still have to run transformations.

        plan = plan_factory()

        output = args.output
        input_ = args.input

        copy_trf = copy(input_, out_arr_t=output)
        copy_comp = PureParallel.from_trf(copy_trf, copy_trf.parameter.input)
        plan.computation_call(copy_comp, output, input_)

        return plan

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        if helpers.product([args.input.shape[i] for i in self._axes]) == 1:
            return self._build_trivial_plan(plan_factory, args)

        plan = plan_factory()

        output = args.output
        input_ = args.input
        inverse = args.inverse

        axes = tuple(sorted(self._axes))
        shape = list(input_.shape)

        if all(shape[axis] % 2 == 0 for axis in axes):
            # If all shift axes have even length, it is possible to perform the shift inplace
            # (by swapping pairs of elements).
            # Note that the inplace fftshift is its own inverse.
            shape[axes[0]] //= 2
            plan.kernel_call(
                TEMPLATE.get_def("fftshift_inplace"),
                [output, input_],
                kernel_name="kernel_fftshift_inplace",
                global_size=shape,
                render_kwds=dict(axes=axes),
            )
        else:
            # Resort to an out-of-place shift to a temporary array and then copy.
            temp = plan.temp_array_like(output)
            plan.kernel_call(
                TEMPLATE.get_def("fftshift_outplace"),
                [temp, input_, inverse],
                kernel_name="kernel_fftshift_outplace",
                global_size=shape,
                render_kwds=dict(axes=axes),
            )

            copy_trf = copy(input_, out_arr_t=output)
            copy_comp = PureParallel.from_trf(copy_trf, copy_trf.parameter.input)
            plan.computation_call(copy_comp, output, temp)

        return plan
