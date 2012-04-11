import numpy
from helpers import *

from tigger.matrixmul import MatrixMul

def test_errors(env, double):

    s1 = (100, 200)
    s2 = (200, 100)
    dtype = numpy.float64 if double else numpy.float32

    a = getTestArray(s1, dtype)
    b = getTestArray(s2, dtype)

    a_dev = env.toDevice(a)
    b_dev = env.toDevice(b)
    dot = MatrixMul(env, debug=True).prepare_for(a_dev, b_dev)
    res_dev = dot(a_dev, b_dev)

    assert diff(env.fromDevice(res_dev), numpy.dot(a, b)) < 1e-6
