import itertools

import numpy
import pytest

from helpers import *
from reikna.reduce import Reduce
from reikna.helpers import template_def
from reikna.cluda import Module
import reikna.cluda.dtypes as dtypes


shapes = [
    (2,), (13,), (1535,), (512 * 231,),
    (140, 3), (13, 598), (1536, 789),
    (5, 15, 19), (134, 25, 23), (145, 56, 178)]
shapes_and_axes = [(shape, axis) for shape, axis in itertools.product(shapes, [None, 0, 1, 2])
    if axis is None or axis < len(shape)]
shapes_and_axes_ids = [str(shape) + "," + str(axis) for shape, axis in shapes_and_axes]


@pytest.mark.parametrize(('shape', 'axis'), shapes_and_axes, ids=shapes_and_axes_ids)
def test_normal(thr, shape, axis):

    rd = Reduce(thr)

    a = get_test_array(shape, numpy.int64)
    a_dev = thr.to_device(a)
    b_ref = a.sum(axis)
    if len(b_ref.shape) == 0:
        b_ref = numpy.array([b_ref], numpy.int64)
    b_dev = thr.array(b_ref.shape, numpy.int64)

    rd.prepare_for(b_dev, a_dev, axis=axis)
    rd(b_dev, a_dev)
    assert diff_is_negligible(b_dev.get(), b_ref)


def test_nondefault_function(thr):
    rd = Reduce(thr)
    shape = (100, 100)
    a = get_test_array(shape, numpy.int64)
    a_dev = thr.to_device(a)
    b_ref = a.sum(0)
    b_dev = thr.array((100,), numpy.int64)

    predicate = lambda output, input: Module(
        template_def(
            ['v1', 'v2'],
            "return ${v1} + ${v2};"),
        snippet=True)

    rd.prepare_for(b_dev, a_dev, axis=0, predicate=predicate)

    rd(b_dev, a_dev)
    assert diff_is_negligible(b_dev.get(), b_ref)
