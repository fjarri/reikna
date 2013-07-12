"""
This module is based on the paper by Salmon et al.,
`P. Int. C. High. Perform. 16 (2011) <http://dx.doi.org/doi:10.1145/2063384.2063405>`_.
and the source code of `Random123 library <http://www.thesalmons.org/john/random123/>`_.

A counter-based random-number generator (CBRNG) is a parametrized function :math:`f_k(c)`,
where :math:`k` is the key, :math:`c` is the counter, and the function :math:`f_k` defines
a bijection in the set of integer numbers.
Being applied to successive counters, the function produces a sequence of pseudo-random numbers.
The key is an analogue of the seed of stateful RNGs;
if the CBRNG is used to generate random num bers in parallel threads, the key is a combination
of a seed and a unique thread number.

There are two types of generators available, ``threefry`` (uses large number of simple functions),
and ``philox`` (uses smaller number of more complicated functions).
The latter one is generally faster on GPUs; see the paper above for detailed comparisons.
These generators can be further specialized to use ``words=2`` or ``words=4``
``bitness=32``-bit or ``bitness=64``-bit counters.
Obviously, the period of the generator equals to the cardinality of the set of possible counters.
For example, if the counter consits of 4 64-bit numbers,
then the period of the generator is :math:`2^{256}`.
As for the key size, in case of ``threefry`` the key has the same size as the counter,
and for ``philox`` the key is half its size.

This implementation sets one of the words of the key (except for ``philox-2x64``,
where 32 bit of the only word in the key are used), the rest are the same for all threads
and are derived from the provided ``seed``.
This limits the maximum number of number-generating threads (``size``).
``philox-2x32`` has a 32-bit key and therefore cannot be used in :py:class:`~reikna.cbrng.CBRNG`
(although it can be used separately as a part of larger kernel).

The :py:class:`~reikna.cbrng.CBRNG` class itself is stateless, same as other computations in Reikna,
so you have to manage the generator state yourself.
The state is created by :py:func:`~reikna.cbrng.create_counters`
and contains either a single counter if the target distribution uses the fixed number
of generated integers, or ``size`` counters otherwise.
This state is then passed to, and updated by a :py:class:`~reikna.cbrng.CBRNG` object.


The following values can be passed as ``distribution`` parameter:

* ``uniform_integer``, parameters: ``min=0``, ``max=2**bitness-1``,
  dtypes: ``(u)int(32/64)``.
  Generates uniformly distributed integer numbers in the interval ``[min, max)``.
  If the size of the interval is a power of 2, a fixed number of counters is used in each thread.

* ``uniform_float``, parameters:  ``min=0``, ``max=1``, dtypes: ``float(32/64)``.
  Generates uniformly distributed floating-points numbers in the interval ``[min, max)``.
  A fixed number of counters is used in each thread.

* ``normal_bm``, parameters: ``mean=0``, ``std=1``,  dtypes: ``float(32/64)``.
  Generates normally distributed random numbers with the mean ``mean`` and
  the standard deviation ``std`` using Box-Muller transform.
  A fixed number of counters is used in each thread.

* ``gamma``, parameters: ``shape=1``, ``scale=1``,  dtypes: ``float(32/64)``.
  Generates random numbers from the gamma distribution

  .. math::
      P(x) = x^{k-1} \\frac{e^{-x/\\theta}}{\\theta^k \\Gamma(k)},

  where :math:`k` is ``shape``, and :math:`\\theta` is ``scale``.
"""

import time
import numpy

from reikna.helpers import *
from reikna.core import *
import reikna.cluda.dtypes as dtypes


TEMPLATE = template_for(__file__)


def create_counters(size, rng, distribution, rng_params):
    """
    Create a counter array on a device for use in :py:class:`~reikna.cbrng.CBRNG`.

    :param size: a shape of the target random numbers array.
    :param rng: random number generator name.
    :param distribution: random distribution name.
    :param rng_params: random number generator parameters.
    """
    size = wrap_in_tuple(size)
    return numpy.zeros(
        size + (rng_params['words'],),
        numpy.uint32 if rng_params['bitness'] == 32 else numpy.uint64)


def create_key(rng, rng_params, seed=None):
    full_key = numpy.zeros(
        rng_params['words'] // (2 if rng == 'philox' else 1),
        numpy.uint32 if rng_params['bitness'] == 32 else numpy.uint64)

    bitness = rng_params['bitness']
    if bitness == 32:
        key_words = full_key.size - 1
    else:
        if full_key.size > 1:
            key_words = (full_key.size - 1) * 2
        else:
            # Philox-2x64 case, key is a single 64-bit integer.
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

    subwords = bitness // 16
    for i, x in enumerate(key):
        full_key[i // subwords] += x << (16 * (subwords - 1 - i % subwords))

    return full_key


class CBRNG(Computation):
    """
    Counter-based pseudo-random number generator class.

    :param new_counters: array of updated counters.
    :param randoms: array with generated random numbers.
    :param old_counters: array of initial counters,
        generated by :py:func:`~reikna.cbrng.create_counters`.
    :param seed: ``None`` for random seed, or an integer.
    :param rng: ``"philox"`` or ``"threefry"``.
    :param rng_params: a dictionary with ``bitness`` (32 or 64,
        corresponds to the size of generated random integers),
        ``words`` (2 or 4, number of integers generated in one go),
        and ``rounds`` (the more rounds, the better randomness is achieved;
        default values are big enough to qualify as PRNG).
    :param distribution: name of the distribution; see the list above.
    :param distribution_params: a dictionary with distribution-specific parameters.
    """

    def __init__(self, randoms_arr, counters_dim, seed=None,
            rng='philox', rng_params=None,
            distribution='int32', distribution_params=None):

        assert rng in ('philox', 'threefry')

        counters_size = randoms_arr.shape[-counters_dim:]
        counters_arr = create_counters(counters_size, rng, distribution, rng_params)

        self._rng = rng
        self._distribution = distribution
        self._counters_dim = counters_dim

        default_rounds = dict(philox=10, threefry=20)[rng]
        rng_params_default = AttrDict(bitness=64, words=4, rounds=default_rounds)
        if rng_params is not None:
            rng_params_default.update(rng_params)
        self._rng_params = rng_params_default
        self._rng_params.key = create_key(self._rng, self._rng_params, seed=seed)

        distribution_params_default = dict(
            uniform_integer=AttrDict(min=0, max=2**self._rng_params.bitness),
            uniform_float=AttrDict(min=0, max=1),
            normal_bm=AttrDict(mean=0, std=1),
            gamma=AttrDict(shape=1, scale=1))
        distribution_params_default = distribution_params_default[distribution]
        if distribution_params is not None:
            distribution_params_default.update(distribution_params)
        self._distribution_params = distribution_params_default

        Computation.__init__(self, [
            Parameter('new_counters', Annotation(counters_arr, 'o')),
            Parameter('randoms', Annotation(randoms_arr, 'o')),
            Parameter('old_counters', Annotation(counters_arr, 'i'))])

    def _build_plan(self, plan_factory, device_params):

        plan = plan_factory()

        plan.kernel_call(
            TEMPLATE.get_def('cbrng'),
            [self.new_counters, self.randoms, self.old_counters],
            global_size=product(self.old_counters.shape[:-1]),
            dependencies=[
                (self.new_counters, self.old_counters),
                (self.new_counters, self.randoms)],
            render_kwds=dict(
                rng=self._rng,
                rng_params=self._rng_params,
                distribution=self._distribution,
                distribution_params=self._distribution_params,
                batch=product(self.randoms.shape[:-self._counters_dim]),
                counters_slices=[self._counters_dim, 1],
                randoms_slices=[
                    len(self.randoms.shape) - self._counters_dim,
                    self._counters_dim]))

        return plan
