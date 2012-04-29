import numpy
import pytest

from helpers import *

from tigger.matrixmul import MatrixMul
import tigger.cluda.dtypes as dtypes

@pytest.mark.parametrize("complex1", [False, True])
@pytest.mark.parametrize("complex2", [False, True])
def test_errors(env, double, complex1, complex2):

    s1 = (100, 200)
    s2 = (200, 100)

    dtype = numpy.float64 if double else numpy.float32
    dtype1 = dtypes.complex_for(dtype) if complex1 else dtype
    dtype2 = dtypes.complex_for(dtype) if complex2 else dtype

    a = getTestArray(s1, dtype1)
    b = getTestArray(s2, dtype2)

    a_dev = env.toDevice(a)
    b_dev = env.toDevice(b)
    dot = MatrixMul(env, debug=True).prepare_for(a_dev, b_dev)
    res_dev = dot(a_dev, b_dev)

    assert diff(env.fromDevice(res_dev), numpy.dot(a, b)) < 1e-6
