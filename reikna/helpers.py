"""Various auxiliary functions which are used throughout the library."""

import collections
import functools
import inspect
import itertools
import os.path
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Generic, Protocol, TypeVar

import numpy
from grunnur import Template


class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...


Node = TypeVar("Node", bound=Comparable)
NewNode = TypeVar("NewNode", bound=Comparable)


def sorted_pair(x: Node, y: Node) -> tuple[Node, Node]:
    return (x, y) if x < y else (y, x)


class Graph(Generic[Node]):
    def __init__(self, pairs: Iterable[tuple[Node, Node]] | None = None):
        self._pairs: set[tuple[Node, Node]] = set()  # each pair is sorted (first < second)
        self._nodes: dict[Node, set[Node]] = collections.defaultdict(set)
        if pairs is not None:
            self.add_edges(pairs)

    def add_edge(self, node1: Node, node2: Node) -> None:
        if node1 == node2:
            raise ValueError("nodes must be distinct")
        self._nodes[node1].add(node2)
        self._nodes[node2].add(node1)
        self._pairs.add(sorted_pair(node1, node2))

    def add_edges(self, pairs: Iterable[tuple[Node, Node]]) -> None:
        for node1, node2 in pairs:
            self.add_edge(node1, node2)

    def add_graph(self, graph: "Graph[Node]") -> None:
        for node1, node2 in graph.pairs():
            self.add_edge(node1, node2)

    def add_cluster(self, nodes: Iterable[Node]) -> None:
        self.add_edges(itertools.combinations(nodes, 2))

    def remove_node(self, node: Node) -> None:
        deps = self._nodes[node]
        for dep in deps:
            self._nodes[dep].remove(node)
            self._pairs.remove(sorted_pair(node, dep))
        del self._nodes[node]

    def remove_edge(self, node1: Node, node2: Node) -> None:
        if node1 == node2:
            raise ValueError("nodes must be distinct")
        self._pairs.remove(sorted_pair(node1, node2))

        self._nodes[node1].remove(node2)
        if len(self._nodes[node1]) == 0:
            del self._nodes[node1]

        self._nodes[node2].remove(node1)
        if len(self._nodes[node2]) == 0:
            del self._nodes[node2]

    def __getitem__(self, node: Node) -> set[Node]:
        return self._nodes[node]

    def pairs(self) -> Iterable[tuple[Node, Node]]:
        return self._pairs

    def translate(self, translator: Callable[[Node], NewNode]) -> "Graph[NewNode]":
        pairs = []
        for node1, node2 in self._pairs:
            pairs.append(sorted_pair(translator(node1), translator(node2)))
        return Graph(pairs)


def product(seq: Iterable[int]) -> int:
    """Returns the product of elements in the iterable ``seq``."""
    return functools.reduce(lambda x1, x2: x1 * x2, seq, 1)


def min_blocks(length: int, block: int) -> int:
    """
    Returns minimum number of blocks with length ``block``
    necessary to cover the array with length ``length``.
    """
    return (length - 1) // block + 1


def log2(num: int) -> int:
    """
    Integer-valued logarigthm with base 2.
    If ``n`` is not a power of 2, the result is rounded to the smallest number.
    """
    pos = 0
    for pow_ in [16, 8, 4, 2, 1]:
        if num >= 2**pow_:
            num //= 2**pow_
            pos += pow_
    return pos


def bounding_power_of_2(num: int) -> int:
    """Returns the minimal number of the form ``2**m`` such that it is greater or equal to ``n``."""
    if num == 1:
        return 1
    result: int = 2 ** (log2(num - 1) + 1)
    return result


def factors(num: int, limit: int | None = None) -> list[tuple[int, int]]:
    """
    Returns the list of pairs ``(factor, num/factor)`` for all factors of ``num``
    (including 1 and ``num``), sorted by ``factor``.
    If ``limit`` is set, only pairs with ``factor <= limit`` are returned.
    """
    if limit is None or limit > num:
        limit = num

    float_sqrt = num**0.5
    int_sqrt = int(round(float_sqrt))

    result = []

    int_limit = 1 + (int_sqrt if int_sqrt**2 == num else int(float_sqrt))

    for i in range(1, int_limit):
        div, mod = divmod(num, i)
        if mod == 0:
            result.append((i, div))

    if limit > result[-1][0]:
        to_rev = result[:-1] if int_sqrt**2 == num else result

        result = result + [(div, f) for f, div in reversed(to_rev)]

    return [r for r in result if r[0] <= limit]


def wrap_in_tuple(seq_or_elem: None | int | Iterable[int]) -> tuple[int, ...]:
    """
    If ``seq_or_elem`` is a sequence, converts it to a ``tuple``,
    otherwise returns a tuple with a single element ``seq_or_elem``.
    """
    if seq_or_elem is None:
        return tuple()
    if isinstance(seq_or_elem, int):
        return (seq_or_elem,)
    return tuple(seq_or_elem)


class IgnoreIntegerOverflow:
    """Context manager for ignoring integer overflow in numpy operations on scalars."""

    def __enter__(self) -> None:
        self._settings = numpy.seterr(over="ignore")

    def __exit__(self, *args: object, **kwds: object) -> None:
        numpy.seterr(**self._settings)


def normalize_axes(ndim: int, axes: None | int | Iterable[int]) -> tuple[int, ...]:
    """
    Transform an iterable of array axes (which can be negative) or a single axis
    into a tuple of non-negative axes.
    """
    if axes is None:
        axes = tuple(range(ndim))
    else:
        axes = wrap_in_tuple(axes)
        axes = tuple(axis if axis >= 0 else ndim + axis for axis in axes)
        if any(axis < 0 or axis >= ndim for axis in axes):
            raise IndexError("Array index out of range")
    return axes


def are_axes_innermost(ndim: int, axes: Sequence[int]) -> bool:
    inner_axes = list(range(ndim - len(axes), ndim))
    return all(axis == inner_axis for axis, inner_axis in zip(axes, inner_axes, strict=False))


def make_axes_innermost(ndim: int, axes: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Given the total number of array axes and a list of axes in this range,
    produce a transposition plan (suitable e.g. for ``numpy.transpose()``)
    that will move make the given axes innermost (in the order they're given).
    Returns the transposition plan, and the plan to transpose the resulting array back
    to the original axes order.
    """
    orig_order = list(range(ndim))
    outer_axes = [i for i in orig_order if i not in axes]
    transpose_to = outer_axes + list(axes)

    transpose_from: list[int] = [0] * ndim  # these values will be all rewritten by construction
    for i, axis in enumerate(transpose_to):
        transpose_from[axis] = i

    return tuple(transpose_to), tuple(transpose_from)
