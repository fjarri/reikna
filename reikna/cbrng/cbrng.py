import numpy

import reikna.helpers as helpers
from reikna.core import Computation, Parameter, Annotation, Type
from reikna.cbrng.tools import KeyGenerator
from reikna.cbrng.bijections import philox
from reikna.cbrng.samplers import SAMPLERS

TEMPLATE = helpers.template_for(__file__)


class CBRNG(Computation):
    """
    Bases: :py:class:`~reikna.core.Computation`

    Counter-based pseudo-random number generator class.

    :param randoms_arr: an array intended for storing generated random numbers.
    :param generators_dim: the number of dimensions (counting from the end)
        which will use independent generators.
        For example, if ``randoms_arr`` has the shape ``(100, 200, 300)`` and
        ``generators_dim`` is ``2``, then in every sub-array ``(j, :, :)``,
        ``j = 0 .. 99``, every element will use an independent generator.
    :param sampler: a :py:class:`~reikna.cbrng.samplers.Sampler` object.
    :param seed: ``None`` for random seed, or an integer.

    .. py:classmethod:: sampler_name(randoms_arr, generators_dim, sampler_kwds=None, seed=None)

        A convenience constructor for the sampler ``sampler_name``
        from :py:mod:`~reikna.cbrng.samplers`.
        The contents of the dictionary ``sampler_kwds`` will be passed to the sampler constructor
        function (with ``bijection`` being created automatically,
        and ``dtype`` taken from ``randoms_arr``).

    .. py:method:: compiled_signature(counters:io, randoms:o)

        :param counters: the RNG "state".
            All attributes are equal to the ones of the result of :py:meth:`create_counters`.
        :param randoms: generated random numbers.
            All attributes are equal to the ones of ``randoms_arr`` from the constructor.
    """

    def __init__(self, randoms_arr, generators_dim, sampler, seed=None):

        self._sampler = sampler
        self._keygen = KeyGenerator.create(sampler.bijection, seed=seed, reserve_id_space=True)

        assert sampler.dtype == randoms_arr.dtype

        counters_size = randoms_arr.shape[-generators_dim:]

        self._generators_dim = generators_dim
        self._counters_t = Type(sampler.bijection.counter_dtype, shape=counters_size)

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
            kernel_name="kernel_cbrng",
            global_size=helpers.product(counters.shape),
            render_kwds=dict(
                sampler=self._sampler,
                keygen=self._keygen,
                batch=helpers.product(randoms.shape[:-self._generators_dim]),
                counters_slices=[self._generators_dim],
                randoms_slices=[
                    len(randoms.shape) - self._generators_dim,
                    self._generators_dim]))

        return plan


# For some reason, closure did not work correctly.
# This class encapsulates the context and provides a classmethod for a given sampler.
class _ConvenienceCtr:

    def __init__(self, sampler_name):
        self._sampler_func = SAMPLERS[sampler_name]

    def __call__(self, cls, randoms_arr, generators_dim, sampler_kwds=None, seed=None):
        bijection = philox(64, 4)
        if sampler_kwds is None:
            sampler_kwds = {}
        sampler = self._sampler_func(bijection, randoms_arr.dtype, **sampler_kwds)
        return cls(randoms_arr, generators_dim, sampler, seed=seed)


# Add convenience constructors to CBRNG
for name in SAMPLERS:
    ctr = _ConvenienceCtr(name)
    setattr(CBRNG, name, classmethod(ctr))
