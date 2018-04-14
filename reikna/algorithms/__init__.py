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

.. autoclass:: Reduce
    :members:

Scan
^^^^

.. autoclass:: Scan
    :members:


Predicates
^^^^^^^^^^

.. autoclass:: Predicate
    :members:

.. autofunction:: predicate_sum
"""

from reikna.algorithms.pureparallel import PureParallel
from reikna.algorithms.transpose import Transpose
from reikna.algorithms.reduce import Reduce
from reikna.algorithms.scan import Scan
from reikna.algorithms.predicates import Predicate, predicate_sum
