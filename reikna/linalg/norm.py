from collections.abc import Callable, Iterable

from grunnur import ArrayMetadata, AsArrayMetadata, DeviceParameters

from ..algorithms import Reduce, predicate_sum
from ..core import Annotation, Computation, ComputationPlan, KernelArguments, Parameter, Type
from ..transformations import norm_const


class EntrywiseNorm(Computation):
    r"""
    Calculates the entrywise matrix norm (same as ``numpy.linalg.norm``)
    of an arbitrary order :math:`r`.

    .. math::

        ||A||_r = \left( \sum_{i,j,\ldots} |A_{i,j,\ldots}|^r \right)^{1 / r}

    :param arr_t: an array-like defining the initial array.
    :param order: the order :math:`r` (any real number).
    :param axes: a list of non-repeating axes to sum over.
        If ``None``, the norm of the whole array will be calculated.

    .. py:method:: compiled_signature(output:o, input:i)

        :param input: an array with the attributes of ``arr_t``.
        :param output: an array with the attributes of ``arr_t``,
            with its shape missing axes from ``axes``.
    """

    def __init__(self, arr_t: AsArrayMetadata, order: float = 2, axes: Iterable[int] | None = None):
        input_ = arr_t.as_array_metadata()

        tr_elems = norm_const(input_, order)
        out_dtype = tr_elems.parameter.output.dtype

        rd = Reduce(Type.array(out_dtype, input_.shape), predicate_sum(out_dtype), axes=axes)

        res_t = rd.parameter.output
        tr_sum = norm_const(res_t, 1.0 / order)

        rd.parameter.input.connect(
            tr_elems, tr_elems.parameter.output, input_prime=tr_elems.parameter.input
        )
        rd.parameter.output.connect(
            tr_sum, tr_sum.parameter.input, output_prime=tr_sum.parameter.output
        )

        self._rd = rd

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(res_t, "o")),
                Parameter("input", Annotation(input_, "i")),
            ],
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        device_params: DeviceParameters,  # noqa: ARG002
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()
        plan.computation_call(self._rd, args.output, args.input)
        return plan
