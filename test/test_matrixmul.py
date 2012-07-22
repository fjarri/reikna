import numpy
import pytest

from helpers import *

from tigger.matrixmul import MatrixMul
from tigger import Transformation
import tigger.cluda.dtypes as dtypes

@pytest.mark.parametrize("complex1", [False, True], ids=['real', 'complex'])
@pytest.mark.parametrize("complex2", [False, True], ids=['real', 'complex'])
def test_errors(ctx_and_double, complex1, complex2):

    ctx, double = ctx_and_double

    s1 = (100, 200)
    s2 = (200, 100)

    dtype = numpy.float64 if double else numpy.float32
    dtype1 = dtypes.complex_for(dtype) if complex1 else dtype
    dtype2 = dtypes.complex_for(dtype) if complex2 else dtype
    res_dtype = numpy.result_type(dtype1, dtype2)

    a = get_test_array(s1, dtype1)
    b = get_test_array(s2, dtype2)

    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    res_dev = ctx.allocate((s1[0], s2[1]), dtype=res_dtype)
    dot = MatrixMul(ctx).prepare_for(res_dev, a_dev, b_dev)
    dot(res_dev, a_dev, b_dev)

    assert diff_is_negligible(ctx.from_device(res_dev), numpy.dot(a, b))
