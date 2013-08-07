"""
Test standard transformations
"""
import pytest

from reikna.cluda import Snippet
from reikna.pureparallel import PureParallel
import reikna.transformations as tr
from reikna.core import Parameter, Annotation, Type

from helpers import *


def pytest_generate_tests(metafunc):
    int_dtypes = [numpy.dtype('int32'), numpy.dtype('int64')]
    float_dtypes = [numpy.dtype('float32')]
    complex_dtypes = [numpy.dtype('complex64')]

    if 'any_dtype' in metafunc.funcargnames:
        dtypes = int_dtypes + float_dtypes + complex_dtypes
        metafunc.parametrize('any_dtype', dtypes, ids=[str(x) for x in dtypes])


def get_test_computation(arr_t):
    return PureParallel(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_idx}(${idxs[0]}, ${input.load_idx}(${idxs[0]}));")


def test_copy(some_thr, any_dtype):

    input_ = get_test_array((1000,), any_dtype)
    input_dev = some_thr.to_device(input_)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    copy = tr.copy(input_dev)

    test.parameter.input.connect(copy, copy.output, input_prime=copy.input)
    test.parameter.output.connect(copy, copy.input, output_prime=copy.output)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), input_)


def test_scale_param(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p1 = get_test_array((1,), any_dtype)[0]
    p2 = get_test_array((1,), any_dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    scale = tr.scale_param(input_dev, any_dtype)

    test.parameter.input.connect(scale, scale.output, input_prime=scale.input, p1=scale.coeff)
    test.parameter.output.connect(scale, scale.input, output_prime=scale.output, p2=scale.coeff)
    testc = test.compile(some_thr)

    testc(output_dev, p1, input_dev, p2)
    assert diff_is_negligible(output_dev.get(), input * p1 * p2)


def test_scale_const(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p1 = get_test_array((1,), any_dtype)[0]
    p2 = get_test_array((1,), any_dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    scale1 = tr.scale_const(input_dev, p1)
    scale2 = tr.scale_const(input_dev, p2)

    test.parameter.input.connect(scale1, scale1.output, input_prime=scale1.input)
    test.parameter.output.connect(scale2, scale2.input, output_prime=scale2.output)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), input * p1 * p2)


def test_split_combine_complex(some_thr):

    i1 = get_test_array((1000,), numpy.float32)
    i2 = get_test_array((1000,), numpy.float32)
    i1_dev = some_thr.to_device(i1)
    i2_dev = some_thr.to_device(i2)
    o1_dev = some_thr.empty_like(i1)
    o2_dev = some_thr.empty_like(i2)

    base_t = Type(numpy.complex64, shape=1000)
    test = get_test_computation(base_t)
    combine = tr.combine_complex(base_t)
    split = tr.split_complex(base_t)

    test.parameter.input.connect(combine, combine.output, i_real=combine.real, i_imag=combine.imag)
    test.parameter.output.connect(split, split.input, o_real=split.real, o_imag=split.imag)
    testc = test.compile(some_thr)

    testc(o1_dev, o2_dev, i1_dev, i2_dev)
    assert diff_is_negligible(o1_dev.get(), i1)
    assert diff_is_negligible(o2_dev.get(), i2)
