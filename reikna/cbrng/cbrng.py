import numpy

import reikna.helpers as helpers
import reikna.cluda.dtypes as dtypes
from reikna.core import Computation, Parameter, Annotation

TEMPLATE = helpers.template_for(__file__)



def create_key(self, rng, words, bitness, seed=None):
    full_key = numpy.zeros(
        rng_words // (2 if rng == 'philox' else 1),
        numpy.uint32 if rng_bitness == 32 else numpy.uint64)

    if rng_bitness == 32:
        key_words = full_key.size - 1
    else:
        if full_key.size > 1:
            key_words = (full_key.size - 1) * 2
        else:
            # Philox-2x64 case, the key is a single 64-bit integer.
            # We use first 32 bit for the key, and the remaining 32 bit for a thread identifier.
            key_words = 1

    if isinstance(seed, numpy.ndarray):
        # explicit key was provided
        assert seed.size == key_words and seed.dtype == numpy.uint32
        key = seed.flatten()
    else:
        # use numpy to generate the key from seed
        np_rng = numpy.random.RandomState(seed)

        # 32-bit Python can only generate random integer up to 2**31-1
        key = np_rng.randint(0, 2**16, key_words * 2)

    subwords = rng_bitness // 16
    for i, key_subword in enumerate(key):
        full_key[i // subwords] += key_subword << (16 * (subwords - 1 - i % subwords))

    return full_key



class CBRNG(Computation):
    """
    Counter-based pseudo-random number generator class.

    :param rng: an object of any RNG class;
        see :py:mod:`~reikna.cbrng` documentation for the list.
    :param distribution: an object of any distribution class;
        see :py:mod:`~reikna.cbrng` documentation for the list.
    :param seed: ``None`` for random seed, or an integer.
    """

    def __init__(self, randoms_arr, counters_dim, distribution, rng=None, seed=None):

        if rng is None:
            rng = Philox(64, 4, seed=seed)
        self._rng_module = rng.module
        self._distribution_module = distribution.module

        counters_size = randoms_arr.shape[-counters_dim:]

        self._counters_dim = counters_dim
        self._counters_t = Type(
            numpy.uint32 if rng.bitness == 32 else numpy.uint64,
            shape=counters_size + (rng.words,))

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
            global_size=helpers.product(old_counters.shape[:-1]),
            render_kwds=dict(
                rng=self._rng_module,
                distribution=self._distribution_module,
                batch=helpers.product(randoms.shape[:-self._counters_dim]),
                counters_slices=[self._counters_dim, 1],
                randoms_slices=[
                    len(randoms.shape) - self._counters_dim,
                    self._counters_dim]))

        return plan
