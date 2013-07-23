import numpy
import pytest

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

    N = 2000
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

    C_ref, D_ref = mock_dummy_nested(A, B, coeff, second_coeff)

    assert diff_is_negligible(C, C_ref)
    assert diff_is_negligible(D, D_ref)


def test_incorrect_parameter_name():
    """
    Tests that setting incorrect parameter name
    in the Computation constructor raises an exception.
    """

    N = 200
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    with pytest.raises(ValueError):
        d = Dummy(A, B, numpy.float32, test_incorrect_parameter_name=True)


def test_untyped_scalar(some_thr):
    """
    Tests that passing an untyped (i.e. not a ndarray with shape==tuple())
    scalar as an argument to a kernel while creating a plan raises an exception.
    """

    N = 200
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    d = Dummy(A, B, numpy.float32, test_untyped_scalar=True)
    with pytest.raises(TypeError):
        dc = d.compile(some_thr)


def test_kernel_adhoc_array(some_thr):
    """
    Tests that passing an array as an ad hoc argument to a kernel
    while creating a plan raises an exception.
    """

    N = 200
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    d = Dummy(A, B, numpy.float32, test_kernel_adhoc_array=True)
    with pytest.raises(ValueError):
        dc = d.compile(some_thr)


def test_computation_adhoc_array(some_thr):
    """
    Tests that passing an array as an ad hoc argument to a computation
    while creating a plan raises an exception.
    """

    N = 200
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    d = DummyNested(A, B, numpy.float32, numpy.float32, test_computation_adhoc_array=True)
    with pytest.raises(ValueError):
        dc = d.compile(some_thr)


def test_computation_incorrect_role(some_thr):
    """
    Tests that passing an array which does not support a required role
    (e.g. a role=='i' for role=='o' parameter) as an argument to a computation
    while creating a plan raises an exception.
    """

    N = 200
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    d = DummyNested(A, B, numpy.float32, numpy.float32, test_computation_incorrect_role=True)
    with pytest.raises(TypeError):
        dc = d.compile(some_thr)


def test_computation_incorrect_type(some_thr):
    """
    Tests that passing an argument with an incorrect type as an argument to a computation
    while creating a plan raises an exception.
    """

    N = 200
    A = get_test_array((N, N), numpy.complex64)
    B = get_test_array(N, numpy.complex64)

    d = DummyNested(A, B, numpy.float32, numpy.float32, test_computation_incorrect_type=True)
    with pytest.raises(TypeError):
        dc = d.compile(some_thr)
