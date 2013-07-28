import numpy

import reikna.helpers as helpers
from reikna.cbrng.tools import KeyGenerator
from reikna.core import Computation, Parameter, Annotation, Type

TEMPLATE = helpers.template_for(__file__)


class CBRNG(Computation):
    """
    Counter-based pseudo-random number generator class.

    :param rng: an object of any RNG class;
        see :py:mod:`~reikna.cbrng` documentation for the list.
    :param distribution: an object of any distribution class;
        see :py:mod:`~reikna.cbrng` documentation for the list.
    :param seed: ``None`` for random seed, or an integer.
    """

    def __init__(self, randoms_arr, counters_dim, sampler, seed=None):

        self._sampler = sampler
        self._keygen = KeyGenerator.create(sampler.bijection, seed=seed, reserve_id_space=True)

        assert sampler.dtype == randoms_arr.dtype

        counters_size = randoms_arr.shape[-counters_dim:]

        self._counters_dim = counters_dim
        self._counters_t = Type(
            sampler.bijection.dtype,
            shape=counters_size + (sampler.bijection.counter_words,))

        Computation.__init__(self, [
            Parameter('counters', Annotation(self._counters_t, 'io')),
            Parameter('randoms', Annotation(randoms_arr, 'o'))])

    def create_counters(self):
        """
        Create a counter array for use in :py:class:`~reikna.cbrng.CBRNG`.
        """
        return numpy.zeros(self._counters_t.shape, self._counters_t.dtype)

    def _build_plan(self, plan_factory, _device_params, counters, randoms):

        plan = plan_factory()

        plan.kernel_call(
            TEMPLATE.get_def('cbrng'),
            [counters, randoms],
            global_size=helpers.product(counters.shape[:-1]),
            render_kwds=dict(
                sampler=self._sampler,
                keygen=self._keygen,
                batch=helpers.product(randoms.shape[:-self._counters_dim]),
                counters_slices=[self._counters_dim, 1],
                randoms_slices=[
                    len(randoms.shape) - self._counters_dim,
                    self._counters_dim]))

        return plan
