import time
import itertools

import numpy
import pytest

from grunnur import Snippet, dtypes, Array

from helpers import *
from reikna.algorithms import Reduce, Predicate, predicate_sum
from reikna.helpers import template_def


shapes = [
    (2,),
    (13,),
    (1535,),
    (512 * 231,),
    (140, 3),
    (13, 598),
    (1536, 789),
    (5, 15, 19),
    (134, 25, 23),
    (145, 56, 178),
]
shapes_and_axes = [
    (shape, axis)
    for shape, axis in itertools.product(shapes, [None, 0, 1, 2])
    if axis is None or axis < len(shape)
]
shapes_and_axes_ids = [str(shape) + "," + str(axis) for shape, axis in shapes_and_axes]


@pytest.mark.parametrize(("shape", "axis"), shapes_and_axes, ids=shapes_and_axes_ids)
def test_normal(queue, shape, axis):
    a = get_test_array(shape, numpy.int64)
    a_dev = Array.from_host(queue, a)

    rd = Reduce(a, predicate_sum(numpy.int64), axes=(axis,) if axis is not None else None)

    b_dev = Array.empty_like(queue.device, rd.parameter.output)
    b_ref = a.sum(axis, keepdims=True)

    rdc = rd.compile(queue.device)
    rdc(queue, b_dev, a_dev)

    assert diff_is_negligible(b_dev.get(queue), b_ref)


def test_nondefault_function(queue):
    shape = (100, 100)
    a = get_test_array(shape, numpy.int64)
    a_dev = Array.from_host(queue, a)
    b_ref = a.sum(0)

    predicate = Predicate(Snippet.from_callable(lambda v1, v2: "return ${v1} + ${v2};"), 0)

    rd = Reduce(a_dev, predicate, axes=(0,))

    b_dev = Array.empty_like(queue.device, rd.parameter.output)

    rdc = rd.compile(queue.device)
    rdc(queue, b_dev, a_dev)

    assert diff_is_negligible(b_dev.get(queue), b_ref)


def test_nonsequential_axes(queue):
    shape = (50, 40, 30, 20)
    a = get_test_array(shape, numpy.int64)
    a_dev = Array.from_host(queue, a)
    b_ref = a.sum(0, keepdims=True).sum(
        2, keepdims=True
    )  # sum over axes 0 and 2 of the initial array

    rd = Reduce(a_dev, predicate_sum(numpy.int64), axes=(0, 2))

    b_dev = Array.empty_like(queue.device, rd.parameter.output)

    rdc = rd.compile(queue.device)
    rdc(queue, b_dev, a_dev)

    assert diff_is_negligible(b_dev.get(queue), b_ref)


def test_structure_type(queue):
    shape = (100, 100)
    dtype = dtypes.align(
        numpy.dtype(
            [
                ("i1", numpy.uint32),
                (
                    "nested",
                    numpy.dtype(
                        [
                            ("v", numpy.uint64),
                        ]
                    ),
                ),
                ("i2", numpy.uint32),
            ]
        )
    )

    a = get_test_array(shape, dtype)
    a_dev = Array.from_host(queue, a)

    # Have to construct the resulting array manually,
    # since numpy cannot reduce arrays with struct dtypes.
    b_ref = numpy.empty(100, dtype)
    b_ref["i1"] = a["i1"].sum(0)
    b_ref["nested"]["v"] = a["nested"]["v"].sum(0)
    b_ref["i2"] = a["i2"].sum(0)

    predicate = Predicate(
        Snippet.from_callable(
            lambda v1, v2: """
            ${ctype} result = ${v1};
            result.i1 += ${v2}.i1;
            result.nested.v += ${v2}.nested.v;
            result.i2 += ${v2}.i2;
            return result;
            """,
            render_globals=dict(ctype=dtypes.ctype(dtype)),
        ),
        numpy.zeros(1, dtype)[0],
    )

    rd = Reduce(a_dev, predicate, axes=(0,))

    b_dev = Array.empty_like(queue.device, rd.parameter.output)

    rdc = rd.compile(queue.device)
    rdc(queue, b_dev, a_dev)
    b_res = b_dev.get(queue)

    assert diff_is_negligible(b_res, b_ref)


@pytest.mark.perf
@pytest.mark.returns("GB/s")
def test_summation(queue):
    perf_size = 2**22
    dtype = numpy.dtype("int64")

    a = get_test_array(perf_size, dtype)
    a_dev = Array.from_host(queue.device, a)

    rd = Reduce(a, predicate_sum(dtype))

    b_dev = Array.empty_like(queue.device, rd.parameter.output)
    b_ref = numpy.array([a.sum()], dtype)

    rdc = rd.compile(queue.device)

    attempts = 10
    times = []
    for i in range(attempts):
        t1 = time.time()
        rdc(queue, b_dev, a_dev)
        queue.synchronize()
        times.append(time.time() - t1)

    assert diff_is_negligible(b_dev.get(queue), b_ref)

    return min(times), perf_size * dtype.itemsize
