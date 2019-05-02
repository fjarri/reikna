import numpy

import reikna.helpers as helpers
from reikna.core import Computation, Parameter, Annotation
import reikna.cluda.dtypes as dtypes
from reikna.algorithms import PureParallel
from reikna.transformations import copy

TEMPLATE = helpers.template_for(__file__)


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

    def __init__(self, arr_t, axes=None):

        Computation.__init__(self, [
            Parameter('output', Annotation(arr_t, 'o')),
            Parameter('input', Annotation(arr_t, 'i')),
            Parameter('inverse', Annotation(numpy.int32), default=0)])

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

    def _build_plan(self, plan_factory, device_params, output, input_, inverse):

        if helpers.product([input_.shape[i] for i in self._axes]) == 1:
            return self._build_trivial_plan(plan_factory, output, input_)

        plan = plan_factory()

        axes = tuple(sorted(self._axes))
        shape = list(input_.shape)

        if all(shape[axis] % 2 == 0 for axis in axes):
        # If all shift axes have even length, it is possible to perform the shift inplace
        # (by swapping pairs of elements).
        # Note that the inplace fftshift is its own inverse.
            shape[axes[0]] //= 2
            plan.kernel_call(
                TEMPLATE.get_def('fftshift_inplace'), [output, input_],
                kernel_name="kernel_fftshift_inplace",
                global_size=shape,
                render_kwds=dict(axes=axes))
        else:
        # Resort to an out-of-place shift to a temporary array and then copy.
            temp = plan.temp_array_like(output)
            plan.kernel_call(
                TEMPLATE.get_def('fftshift_outplace'), [temp, input_, inverse],
                kernel_name="kernel_fftshift_outplace",
                global_size=shape,
                render_kwds=dict(axes=axes))

            copy_trf = copy(input_, out_arr_t=output)
            copy_comp = PureParallel.from_trf(copy_trf, copy_trf.input)
            plan.computation_call(copy_comp, output, temp)

        return plan
