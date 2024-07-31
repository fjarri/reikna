"""
This module contains various auxiliary functions which are used throughout the library.
"""

import collections
import functools
import itertools
import sys

if sys.version_info[0] >= 3:
    from collections.abc import Iterable
else:
    from collections import Iterable
import inspect
import os.path
import warnings

from grunnur import Template


class Graph:
    def __init__(self, pairs=None):
        self._pairs = set()
        self._nodes = collections.defaultdict(set)
        if pairs is not None:
            self.add_edges(pairs)

    def add_edge(self, node1, node2):
        assert node1 != node2
        self._nodes[node1].add(node2)
        self._nodes[node2].add(node1)
        self._pairs.add(tuple(sorted((node1, node2))))

    def add_edges(self, pairs):
        for node1, node2 in pairs:
            self.add_edge(node1, node2)

    def add_graph(self, graph):
        for node1, node2 in graph.pairs():
            self.add_edge(node1, node2)

    def add_cluster(self, nodes):
        self.add_edges(itertools.combinations(nodes, 2))

    def remove_node(self, node):
        deps = self._nodes[node]
        for dep in deps:
            self._nodes[dep].remove(node)
            self._pairs.remove(tuple(sorted((node, dep))))
        del self._nodes[node]

    def remove_edge(self, node1, node2):
        assert node1 != node2
        self._pairs.remove(tuple(sorted((node1, node2))))

        self._nodes[node1].remove(node2)
        if len(self._nodes[node1]) == 0:
            del self._nodes[node1]

        self._nodes[node2].remove(node1)
        if len(self._nodes[node2]) == 0:
            del self._nodes[node2]

    def __getitem__(self, node):
        return self._nodes[node]

    def pairs(self):
        return self._pairs

    def translate(self, translator):
        pairs = []
        for node1, node2 in self._pairs:
            pairs.append(tuple(sorted((translator(node1), translator(node2)))))
        return Graph(pairs)


def product(seq):
    """
    Returns the product of elements in the iterable ``seq``.
    """
    return functools.reduce(lambda x1, x2: x1 * x2, seq, 1)


def extract_signature_and_value(func_or_str, default_parameters=None):
    if not inspect.isfunction(func_or_str):
        if default_parameters is None:
            parameters = []
        else:
            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
            parameters = [inspect.Parameter(name, kind=kind) for name in default_parameters]

        return inspect.Signature(parameters), func_or_str

    signature = inspect.signature(func_or_str)

    # pass mock values to extract the value
    args = [None] * len(signature.parameters)
    return signature, func_or_str(*args)


def template_def(signature, code):
    """
    Returns a ``Mako`` template with the given ``signature``.

    :param signature: a list of postitional argument names,
        or a ``Signature`` object from ``inspect`` module.
    :code: a body of the template.
    """
    if not isinstance(signature, inspect.Signature):
        # treating ``signature`` as a list of positional arguments
        # HACK: Signature or Parameter constructors are not documented.
        kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        signature = inspect.Signature([inspect.Parameter(name, kind=kind) for name in signature])

    template_src = "<%def name='_func" + str(signature) + "'>\n" + code + "\n</%def>"
    return Template.from_string(template_src).get_def("_func")


def min_blocks(length, block):
    """
    Returns minimum number of blocks with length ``block``
    necessary to cover the array with length ``length``.
    """
    return (length - 1) // block + 1


def log2(num):
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


def bounding_power_of_2(num):
    """
    Returns the minimal number of the form ``2**m`` such that it is greater or equal to ``n``.
    """
    if num == 1:
        return 1
    else:
        return 2 ** (log2(num - 1) + 1)


def factors(num, limit=None):
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

    if int_sqrt**2 == num:
        int_limit = int_sqrt + 1
    else:
        int_limit = int(float_sqrt) + 1

    for i in range(1, int_limit):
        div, mod = divmod(num, i)
        if mod == 0:
            result.append((i, div))

    if limit > result[-1][0]:
        if int_sqrt**2 == num:
            to_rev = result[:-1]
        else:
            to_rev = result

        result = result + [(div, f) for f, div in reversed(to_rev)]

    return [r for r in result if r[0] <= limit]


def wrap_in_tuple(seq_or_elem):
    """
    If ``seq_or_elem`` is a sequence, converts it to a ``tuple``,
    otherwise returns a tuple with a single element ``seq_or_elem``.
    """
    if seq_or_elem is None:
        return tuple()
    elif isinstance(seq_or_elem, str):
        return (seq_or_elem,)
    elif isinstance(seq_or_elem, Iterable):
        return tuple(seq_or_elem)
    else:
        return (seq_or_elem,)


class ignore_integer_overflow:
    """
    Context manager for ignoring integer overflow in numpy operations on scalars
    (not ignored by default because of a bug in numpy).
    """

    def __init__(self):
        self.catch = warnings.catch_warnings()

    def __enter__(self):
        self.catch.__enter__()
        warnings.filterwarnings("ignore", "overflow encountered in scalar add")

    def __exit__(self, *args, **kwds):
        self.catch.__exit__(*args, **kwds)


def normalize_axes(ndim, axes):
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


def are_axes_innermost(ndim, axes):
    inner_axes = list(range(ndim - len(axes), ndim))
    return all(axis == inner_axis for axis, inner_axis in zip(axes, inner_axes))


def make_axes_innermost(ndim, axes):
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

    transpose_from = [None] * ndim
    for i, axis in enumerate(transpose_to):
        transpose_from[axis] = i

    return tuple(transpose_to), tuple(transpose_from)
