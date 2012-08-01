import itertools

import numpy
import pytest

from helpers import *
from tigger.reduce import Reduce
import tigger.cluda.dtypes as dtypes

output_sizes = [1, 13, 128, 1535, 2048]
multipliers = [140, 113 * 117, 512 * 231, 512 * 512 + 150]
reduce_pairs = [(output_size, multiplier)
    for output_size, multiplier in itertools.product(output_sizes, multipliers)
    if output_size * multiplier <= 2 ** 20]

@pytest.mark.parametrize(
    ('output_size', 'multiplier'), reduce_pairs,
    ids=[str(x) + "x" + str(y) for x, y in reduce_pairs])
def test_normal(ctx, output_size, multiplier):

    rd = Reduce(ctx, "return val1 + val2;")

    input_size = multiplier * output_size

    a = get_test_array(input_size, numpy.int64)
    a_dev = ctx.to_device(a)
    b_dev = ctx.allocate(output_size, numpy.int64)

    rd.prepare(dtype=numpy.int64, input_size=input_size, output_size=output_size)
    rd(b_dev, a_dev)
    assert diff_is_negligible(ctx.from_device(b_dev), a.reshape(output_size, multiplier).sum(1))

    rd.prepare_for(b_dev, a_dev)
    rd(b_dev, a_dev)
    assert diff_is_negligible(ctx.from_device(b_dev), a.reshape(output_size, multiplier).sum(1))
