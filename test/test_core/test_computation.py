import numpy

from helpers import *
from test_core.dummy import *


def test_dummy(thr):

    N = 200
    coeff = 2
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    A_dev = thr.to_device(A)
    B_dev = thr.to_device(B)
    C_dev = thr.empty_like(A_dev)
    D_dev = thr.empty_like(B_dev)

    d = Dummy(A_dev, B_dev, numpy.float32).compile(thr)
    d(C_dev, D_dev, A_dev, B_dev, coeff)
    C = C_dev.get()
    D = D_dev.get()

    C_ref, D_ref = mock_dummy(A, B, coeff)
    assert diff_is_negligible(C, C_ref)
    assert diff_is_negligible(D, D_ref)


def test_nested_dummy(thr):

    N = 200
    coeff = 2
    second_coeff = 3
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    A_dev = thr.to_device(A)
    B_dev = thr.to_device(B)
    C_dev = thr.empty_like(A_dev)
    D_dev = thr.empty_like(B_dev)

    d = DummyNested(A_dev, B_dev, numpy.float32, second_coeff).compile(thr)
    d(C_dev, D_dev, A_dev, B_dev, coeff)
    C = C_dev.get()
    D = D_dev.get()

    C_ref, D_ref = mock_dummy(A, B, coeff)
    C_ref, D_ref = mock_dummy(C_ref, D_ref, second_coeff)

    assert diff_is_negligible(C, C_ref)
    assert diff_is_negligible(D, D_ref)
