import numpy
import pytest

from helpers import *

from tigger.reduce import Reduce
import tigger.cluda.dtypes as dtypes

def test_errors(ctx):

    rd = Reduce(ctx, "return val1 + val2;")

    output_size = 117
    multiplier = 113
    input_size = multiplier * output_size

    a = get_test_array(input_size, numpy.float32)
    a_dev = ctx.to_device(a)
    b_dev = ctx.allocate(output_size, numpy.float32)

    rd.prepare(dtype=numpy.float32, input_size=input_size, output_size=output_size)
    rd(b_dev, a_dev)
    assert diff_is_negligible(ctx.from_device(b_dev), a.reshape(output_size, multiplier).sum(1))

    rd.prepare_for(b_dev, a_dev)
    rd(b_dev, a_dev)
    assert diff_is_negligible(ctx.from_device(b_dev), a.reshape(output_size, multiplier).sum(1))
