from reikna.core import Computation, Parameter, Annotation, Type
from reikna.transformations import norm_const
from reikna.algorithms import Reduce, predicate_sum


class EntrywiseNorm(Computation):
    r"""
    Bases: :py:class:`~reikna.core.Computation`

    Calculates the entrywise matrix norm (same as ``numpy.linalg.norm``)
    of an arbitrary order :math:`r`:

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

    def __init__(self, arr_t, order=2, axes=None):
        tr_elems = norm_const(arr_t, order)
        out_dtype = tr_elems.output.dtype

        rd = Reduce(Type(out_dtype, arr_t.shape), predicate_sum(out_dtype), axes=axes)

        res_t = rd.parameter.output
        tr_sum = norm_const(res_t, 1. / order)

        rd.parameter.input.connect(tr_elems, tr_elems.output, input_prime=tr_elems.input)
        rd.parameter.output.connect(tr_sum, tr_sum.input, output_prime=tr_sum.output)

        self._rd = rd

        Computation.__init__(self, [
            Parameter('output', Annotation(res_t, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()
        plan.computation_call(self._rd, output, input_)
        return plan
