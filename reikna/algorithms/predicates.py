from typing import Any, Callable

import numpy
from grunnur import Snippet
from numpy.typing import DTypeLike


class Predicate:
    """
    A predicate used in some of Reikna algorithms
    (e.g. :py:class:`~reikna.algorithms.Reduce` or :py:class:`~reikna.algorithms.Scan`).

    :param operation: a :py:class:`grunnur.Snippet` object with two parameters
        which will take the names of two arguments to join.
    :param empty: a numpy scalar with the empty value of the argument
        (the one which, being joined by another argument, does not change it).
    """

    def __init__(self, operation: Snippet, empty: Any):
        self.operation = operation
        self.empty = empty


def predicate_sum(dtype: DTypeLike) -> Predicate:
    """
    Returns a :py:class:`~reikna.algorithms.Predicate` object which sums its arguments.
    """
    return Predicate(
        Snippet.from_callable(lambda v1, v2: "return ${v1} + ${v2};"), numpy.zeros(1, dtype)[0]
    )
