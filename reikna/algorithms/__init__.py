"""
General purpose algorithms.


Pure parallel computations
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PureParallel
    :members:


Transposition (permutation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Transpose
    :members:


Reduction
^^^^^^^^^

.. autoclass:: Predicate
    :members:

.. autofunction:: predicate_sum

.. autoclass:: Reduce
    :members:
"""

from reikna.algorithms.pureparallel import PureParallel
from reikna.algorithms.transpose import Transpose
from reikna.algorithms.reduce import Reduce, Predicate, predicate_sum
