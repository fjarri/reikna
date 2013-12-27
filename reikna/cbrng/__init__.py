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

The :py:class:`~reikna.cbrng.CBRNG` class sets one of the words of the key
(except for ``philox-2x64``, where 32 bit of the only word in the key are used),
the rest are the same for all threads and are derived from the provided ``seed``.
This limits the maximum number of number-generating threads (``size``).
``philox-2x32`` has a 32-bit key and therefore cannot be used in :py:class:`~reikna.cbrng.CBRNG`
(although it can be used separately with the help of the kernel API).

The :py:class:`~reikna.cbrng.CBRNG` class itself is stateless, same as other computations in Reikna,
so you have to manage the generator state yourself.
The state is created by the :py:meth:`~reikna.cbrng.CBRNG.create_counters` method
and contains a ``size`` counters.
This state is then passed to, and updated by a :py:class:`~reikna.cbrng.CBRNG` object.


.. autoclass:: CBRNG
    :members:


Kernel API
^^^^^^^^^^

.. automodule:: reikna.cbrng.bijections
    :members:

.. automodule:: reikna.cbrng.samplers
    :members:

.. automodule:: reikna.cbrng.tools
    :members:
"""

from reikna.cbrng.cbrng import CBRNG
