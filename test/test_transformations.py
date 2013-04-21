"""
Test standard transformations
"""
import pytest

from reikna.helpers import template_func
from reikna.cluda import Module
from reikna.elementwise import Elementwise, specialize_elementwise
import reikna.transformations as tr

from helpers import *


def pytest_generate_tests(metafunc):
    int_dtypes = [numpy.dtype('int32'), numpy.dtype('int64')]
    float_dtypes = [numpy.dtype('float32')]
    complex_dtypes = [numpy.dtype('complex64')]

    if 'any_dtype' in metafunc.funcargnames:
        dtypes = int_dtypes + float_dtypes + complex_dtypes
        metafunc.parametrize('any_dtype', dtypes, ids=[str(x) for x in dtypes])


identity = lambda output, input: Module(
    template_func(
        ['output', 'input'],
        """
        ${output.store}(idx, ${input.load}(idx));
        """),
    snippet=True)

TestComputation = specialize_elementwise('output', 'input', None, identity)


def test_identity(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = TestComputation(some_thr)
    test.connect(tr.identity(), 'input', ['input_prime'])
    test.connect(tr.identity(), 'output', ['output_prime'])
    test.prepare_for(output_dev, input_dev)

    test(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), input)


def test_scale_param(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p1 = get_test_array((1,), any_dtype)
    p2 = get_test_array((1,), any_dtype)
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = TestComputation(some_thr)
    test.connect(tr.scale_param(), 'input', ['input_prime'], ['p1'])
    test.connect(tr.scale_param(), 'output', ['output_prime'], ['p2'])
    test.prepare_for(output_dev, input_dev, p1[0], p2[0])

    test(output_dev, input_dev, p1[0], p2[0])
    assert diff_is_negligible(output_dev.get(), input * p1[0] * p2[0])


def test_scale_const(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p = get_test_array((1,), any_dtype)
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = TestComputation(some_thr)
    test.connect(tr.scale_const(p[0]), 'input', ['input_prime'])
    test.prepare_for(output_dev, input_dev)

    test(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), input * p[0])

def test_split_combine_complex(some_thr):

    i1 = get_test_array((1000,), numpy.float32)
    i2 = get_test_array((1000,), numpy.float32)
    i1_dev = some_thr.to_device(i1)
    i2_dev = some_thr.to_device(i2)
    o1_dev = some_thr.empty_like(i1)
    o2_dev = some_thr.empty_like(i2)

    test = TestComputation(some_thr)
    test.connect(tr.combine_complex(), 'input', ['i1', 'i2'])
    test.connect(tr.split_complex(), 'output', ['o1', 'o2'])
    test.prepare_for(o1_dev, o2_dev, i1_dev, i2_dev)

    test(o1_dev, o2_dev, i1_dev, i2_dev)
    assert diff_is_negligible(o1_dev.get(), i1)
    assert diff_is_negligible(o2_dev.get(), i2)
