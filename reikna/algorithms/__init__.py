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

from .predicates import Predicate, predicate_sum
from .pureparallel import PureParallel
from .reduce import Reduce
from .scan import Scan
from .transpose import Transpose
