"""
This module contains various auxiliary functions which are used throughout the library.
"""

from __future__ import division

import functools
import collections
import os.path
import warnings

from mako.template import Template


class AttrDict(dict):
    """
    An extension of the standard ``dict`` class
    which allows one to address its elements as attributes
    (for example, ``d.key`` instead of ``d['key']``).
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __repr__(self):
        return "AttrDict(" + \
            ", ".join((key + "=" + repr(value)) for key, value in self.items()) + ")"


def product(seq):
    """
    Returns the product of elements in the iterable ``seq``.
    """
    return functools.reduce(lambda x1, x2: x1 * x2, seq, 1)


def template_source_for(filename):
    """
    Returns the contents of the file which has the same name as ``filename``
    and the extension ``.mako``.
    Typically used in computation modules as ``template_source_for(__filename__)``.
    """
    name, ext = os.path.splitext(filename)
    return open(name + ".mako").read()


def template_from(template):
    """
    Creates a Mako template object from a given string.
    If ``template`` already has ``render()`` method, does nothing.
    """
    if hasattr(template, 'render'):
        return template
    else:
        return Template(template)


def template_defs_for_code(code, argnames):
    """
    Returns the template source with two Mako defs ``code_functions`` and ``code_kernel``,
    taking parameters with names from ``argnames``
    and containing template snippets from ``code['kernel']`` and ``code['functions']`` (optional).
    """
    return (
        "<%def name='code_functions(" + ", ".join(argnames) + ")'>\n" +
        code.get('functions', "") +
        "\n</%def>" +
        "<%def name='code_kernel(" + ", ".join(argnames) + ")'>\n" +
        code['kernel'] +
        "\n</%def>")


def template_for(filename):
    """
    Returns the Mako template object created from the file
    which has the same name as ``filename`` and the extension ``.mako``.
    Typically used in computation modules as ``template_for(__filename__)``.
    """
    return template_from(template_source_for(filename))


def min_blocks(length, block):
    """
    Returns minimum number of blocks with length ``block``
    necessary to cover the array with length ``length``.
    """
    return (length - 1) // block + 1


def log2(n):
    """
    Integer-valued logarigthm with base 2.
    If ``n`` is not a power of 2, the result is rounded to the smallest number.
    """
    pos = 0
    for pow in [16, 8, 4, 2, 1]:
        if n >= 2 ** pow:
            n //= (2 ** pow)
            pos += pow
    return pos


def bounding_power_of_2(n):
    """
    Returns closest number of the form ``2**m`` such it is greater or equal to ``n``.
    """
    return 2 ** (log2(n - 1) + 1)


def factors(n, limit=None):
    """
    Returns the list of pairs ``(factor, n/factor)`` for all factors of ``n``
    (including 1 and ``n``), sorted by ``factor``.
    If ``limit`` is set, only pairs with ``factor <= limit`` are returned.
    """
    if limit is None:
        limit = n

    result = []
    for i in range(1, min(limit, int(n ** 0.5) + 1)):
        div, mod = divmod(n, i)
        if mod == 0:
            result.append((i, div))

    if limit > result[-1][0]:
        result = result + [(div, f) for f, div in reversed(result)]
        return [r for r in result if r[0] <= limit]
    else:
        return result


def wrap_in_tuple(x):
    """
    If ``x`` is a sequence, converts it to a ``tuple``,
    otherwise returns a tuple with a single element ``x``.
    """
    if x is None:
        return tuple()
    elif isinstance(x, str):
        return (x,)
    elif isinstance(x, collections.Iterable):
        return tuple(x)
    else:
        return (x,)


class ignore_integer_overflow():
    """
    Context manager for ignoring integer overflow in numpy operations on scalars
    (not ignored by default because of a bug in numpy).
    """

    def __init__(self):
        self.catch = warnings.catch_warnings()

    def __enter__(self):
        self.catch.__enter__()
        warnings.filterwarnings("ignore", "overflow encountered in uint_scalars")
        warnings.filterwarnings("ignore", "overflow encountered in ulong_scalars")

    def __exit__(self, *args, **kwds):
        self.catch.__exit__(*args, **kwds)
