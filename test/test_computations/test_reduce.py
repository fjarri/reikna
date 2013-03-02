import itertools

import numpy
import pytest

from helpers import *
from reikna.reduce import Reduce
import reikna.cluda.dtypes as dtypes


shapes = [
    (2,), (13,), (1535,), (512 * 231,),
    (140, 3), (13, 598), (1536, 789),
    (5, 15, 19), (134, 25, 23), (145, 56, 178)]
shapes_and_axes = [(shape, axis) for shape, axis in itertools.product(shapes, [None, 0, 1, 2])
    if axis is None or axis < len(shape)]
shapes_and_axes_ids = [str(shape) + "," + str(axis) for shape, axis in shapes_and_axes]


@pytest.mark.parametrize(('shape', 'axis'), shapes_and_axes, ids=shapes_and_axes_ids)
def test_normal(ctx, shape, axis):

    rd = Reduce(ctx)

    a = get_test_array(shape, numpy.int64)
    a_dev = ctx.to_device(a)
    b_ref = a.sum(axis)
    if len(b_ref.shape) == 0:
        b_ref = numpy.array([b_ref], numpy.int64)
    b_dev = ctx.array(b_ref.shape, numpy.int64)

    rd.prepare_for(b_dev, a_dev, axis=axis,
        code=dict(kernel="return input1 + input2;"))
    rd(b_dev, a_dev)
    assert diff_is_negligible(b_dev.get(), b_ref)


def test_nontrivial_function(ctx):
    rd = Reduce(ctx)
    shape = (100, 100)
    a = get_test_array(shape, numpy.int64)
    a_dev = ctx.to_device(a)
    b_ref = a.sum(0)
    b_dev = ctx.array((100,), numpy.int64)

    rd.prepare_for(b_dev, a_dev, axis=0,
        code=dict(
            kernel="return test(input1, input2);",
            functions="""
            WITHIN_KERNEL ${output.ctype} test(${input.ctype} val1, ${input.ctype} val2)
            {
                return val1 + val2;
            }
            """))

    rd(b_dev, a_dev)
    assert diff_is_negligible(b_dev.get(), b_ref)
