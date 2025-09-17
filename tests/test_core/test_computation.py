import re

import numpy
import pytest
from grunnur import Array, Queue

from helpers import diff_is_negligible, get_test_array
from test_core.dummy import (
    Dummy,
    DummyAdvanced,
    DummyNested,
    mock_dummy,
    mock_dummy_advanced,
    mock_dummy_nested,
)


def test_dummy(queue):
    size = 200
    coeff = 2
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    c_dev = Array.empty(queue.device, a_dev.shape, a_dev.dtype)
    d_dev = Array.empty(queue.device, b_dev.shape, b_dev.dtype)

    dummy = Dummy(a_dev, b_dev, numpy.float32).compile(queue.device)
    dummy(queue, c_dev, d_dev, a_dev, b_dev, coeff)
    c = c_dev.get(queue)
    d = d_dev.get(queue)

    c_ref, d_ref = mock_dummy(a, b, coeff)
    assert diff_is_negligible(c, c_ref)
    assert diff_is_negligible(d, d_ref)


def test_nested_dummy(queue):
    size = 2000
    coeff = 2
    second_coeff = 3
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    c_dev = Array.empty(queue.device, a_dev.shape, a_dev.dtype)
    d_dev = Array.empty(queue.device, b_dev.shape, b_dev.dtype)

    dummy = DummyNested(a_dev, b_dev, numpy.float32, second_coeff).compile(queue.device)
    dummy(queue, c_dev, d_dev, a_dev, b_dev, coeff)
    c = c_dev.get(queue)
    d = d_dev.get(queue)

    c_ref, d_ref = mock_dummy_nested(a, b, coeff, second_coeff)

    assert diff_is_negligible(c, c_ref)
    assert diff_is_negligible(d, d_ref)


def test_incorrect_parameter_name(some_queue):
    """
    Tests that setting incorrect parameter name
    in the Computation constructor raises an exception.
    """
    size = 200
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)
    a_dev = Array.from_host(some_queue, a)
    b_dev = Array.from_host(some_queue, b)

    with pytest.raises(
        ValueError, match=re.escape("External parameter name cannot start with the underscore.")
    ):
        Dummy(a_dev, b_dev, numpy.float32, test_incorrect_parameter_name=True)


def test_untyped_scalar(some_queue):
    """
    Tests that passing an untyped (i.e. not a ndarray with shape==tuple())
    scalar as an argument to a kernel while creating a plan raises an exception.
    """
    size = 200
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)
    a_dev = Array.from_host(some_queue, a)
    b_dev = Array.from_host(some_queue, b)

    dummy = Dummy(a_dev, b_dev, numpy.float32, test_untyped_scalar=True)
    with pytest.raises(TypeError, match="Unknown argument type: <class 'int'>"):
        dummy.compile(some_queue.device)


def test_kernel_adhoc_array(some_queue):
    """
    Tests that passing an array as an ad hoc argument to a kernel
    while creating a plan raises an exception.
    """
    size = 200
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)
    a_dev = Array.from_host(some_queue, a)
    b_dev = Array.from_host(some_queue, b)

    dummy = Dummy(a_dev, b_dev, numpy.float32, test_kernel_adhoc_array=True)
    with pytest.raises(ValueError, match="Arrays are not allowed as ad hoc arguments"):
        dummy.compile(some_queue.device)


def test_computation_adhoc_array(some_queue):
    """
    Tests that passing an array as an ad hoc argument to a computation
    while creating a plan raises an exception.
    """
    size = 200
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)
    a_dev = Array.from_host(some_queue, a)
    b_dev = Array.from_host(some_queue, b)

    dummy = DummyNested(a_dev, b_dev, numpy.float32, 3, test_computation_adhoc_array=True)
    with pytest.raises(ValueError, match="Ad hoc arguments are only allowed for scalar parameters"):
        dummy.compile(some_queue.device)


def test_computation_incorrect_role(some_queue):
    """
    Tests that passing an array which does not support a required role
    (e.g. a role=='i' for role=='o' parameter) as an argument to a computation
    while creating a plan raises an exception.
    """
    size = 200
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)
    a_dev = Array.from_host(some_queue, a)
    b_dev = Array.from_host(some_queue, b)

    dummy = DummyNested(a_dev, b_dev, numpy.float32, 3, test_computation_incorrect_role=True)
    message = re.escape(
        "Got Annotation(ArrayMetadata(dtype=complex64, shape=(200,)), role=i) for 'D', "
        "expected Annotation(ArrayMetadata(dtype=complex64, shape=(200,)), role=o)"
    )
    with pytest.raises(ValueError, match=message):
        dummy.compile(some_queue.device)


def test_computation_incorrect_type(some_queue):
    """
    Tests that passing an argument with an incorrect type as an argument to a computation
    while creating a plan raises an exception.
    """
    size = 200
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)
    a_dev = Array.from_host(some_queue, a)
    b_dev = Array.from_host(some_queue, b)

    dummy = DummyNested(a_dev, b_dev, numpy.float32, 3, test_computation_incorrect_type=True)
    message = re.escape(
        "Got Annotation(ArrayMetadata(dtype=complex64, shape=(200,)), role=i) for 'A_prime', "
        "expected Annotation(ArrayMetadata(dtype=complex64, shape=(200, 200)), role=i)"
    )
    with pytest.raises(ValueError, match=message):
        dummy.compile(some_queue.device)


def test_same_arguments(some_queue):
    """Tests passing the same KernelArgument for two parameters of a computation or a kernel."""
    size = 200
    coeff1 = 3
    coeff2 = 4
    c = get_test_array((size, size), numpy.complex64)
    d = get_test_array((size, size), numpy.complex64)
    c_dev = Array.from_host(some_queue, c)
    d_dev = Array.from_host(some_queue, d)

    dummy = DummyAdvanced(c_dev, numpy.float32)
    dc = dummy.compile(some_queue.device)

    dc(some_queue, c_dev, d_dev, coeff1, coeff2)
    c_ref, d_ref = mock_dummy_advanced(c, d, coeff1, coeff2)

    c = c_dev.get(some_queue)
    d = d_dev.get(some_queue)

    assert diff_is_negligible(c, c_ref)
    assert diff_is_negligible(d, d_ref)


def test_same_arg_as_i_and_o(some_queue):
    """
    Tests that the same 'io' array can be used both as an input and as an output argument
    of the same nested computation.
    """
    size = 2000
    coeff = 2
    second_coeff = 3
    a = get_test_array((size, size), numpy.complex64)
    b = get_test_array(size, numpy.complex64)

    a_dev = Array.from_host(some_queue, a)
    b_dev = Array.from_host(some_queue, b)
    c_dev = Array.empty(some_queue.device, a_dev.shape, a_dev.dtype)
    d_dev = Array.empty(some_queue.device, b_dev.shape, b_dev.dtype)

    dummy = DummyNested(
        a_dev, b_dev, numpy.float32, second_coeff, test_same_arg_as_i_and_o=True
    ).compile(some_queue.device)
    dummy(some_queue, c_dev, d_dev, a_dev, b_dev, coeff)
    c = c_dev.get(some_queue)
    d = d_dev.get(some_queue)

    c_ref, d_ref = mock_dummy_nested(a, b, coeff, second_coeff, test_same_arg_as_i_and_o=True)

    assert diff_is_negligible(c, c_ref)
    assert diff_is_negligible(d, d_ref)
