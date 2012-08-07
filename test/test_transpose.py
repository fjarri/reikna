import itertools

import numpy
import pytest

from helpers import *

from tigger.transpose import Transpose
import tigger.cluda.dtypes as dtypes

lengths = [13, 128, 511, 2049]
matrix_sizes = [(height, width)
    for height, width in itertools.product(lengths, lengths)
    if height * width <= 2 ** 20]

@pytest.mark.parametrize(
    ('height', 'width'), matrix_sizes,
    ids=[str(x) + "x" + str(y) for x, y in matrix_sizes])
def test_errors(ctx, height, width):
    a = get_test_array((height, width), numpy.int32)
    a_dev = ctx.to_device(a)
    res_dev = ctx.allocate((width, height), dtype=numpy.int32)
    dot = Transpose(ctx).prepare_for(res_dev, a_dev)
    dot(res_dev, a_dev)

    assert diff_is_negligible(ctx.from_device(res_dev), a.T)
