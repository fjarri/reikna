import numpy

import reikna.helpers as helpers
from reikna.core import Computation, Parameter, Annotation
import reikna.cluda.dtypes as dtypes
from reikna.algorithms import PureParallel
from reikna.transformations import copy

TEMPLATE = helpers.template_for(__file__)


class FFTShift(Computation):

    def __init__(self, arr_t, axes=None):

        Computation.__init__(self, [
            Parameter('output', Annotation(arr_t, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

        if axes is None:
            axes = tuple(range(len(arr_t.shape)))
        else:
            axes = tuple(axes)
        self._axes = axes

    def _build_trivial_plan(self, plan_factory, output, input_):
        # Trivial problem. Need to add a dummy kernel
        # because we still have to run transformations.

        plan = plan_factory()

        copy_trf = copy(input_, out_arr_t=output)
        copy_comp = PureParallel.from_trf(copy_trf, copy_trf.input)
        plan.computation_call(copy_comp, output, input_)

        return plan

    def _build_plan(self, plan_factory, device_params, output, input_):

        if helpers.product([input_.shape[i] for i in self._axes]) == 1:
            return self._build_trivial_plan(plan_factory, output, input_)

        plan = plan_factory()

        axes = tuple(sorted(self._axes))

        shape = list(input_.shape)
        shape[axes[0]] //= 2

        plan.kernel_call(
            TEMPLATE.get_def('fftshift'), [output, input_],
            global_size=shape,
            render_kwds=dict(axes=axes))

        return plan
