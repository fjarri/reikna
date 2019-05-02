"""
Test standard transformations
"""
import pytest

from reikna.cluda import Snippet
from reikna.algorithms import PureParallel
import reikna.transformations as tr
from reikna.core import Parameter, Annotation, Type

from helpers import *


def pytest_generate_tests(metafunc):
    int_dtypes = [numpy.dtype('int32'), numpy.dtype('int64')]
    real_dtypes = [numpy.dtype('float32')]
    complex_dtypes = [numpy.dtype('complex64')]

    if 'any_dtype' in metafunc.funcargnames:
        dtypes = int_dtypes + real_dtypes + complex_dtypes
        metafunc.parametrize('any_dtype', dtypes, ids=[str(x) for x in dtypes])

    if 'rc_dtype' in metafunc.funcargnames:
        dtypes = real_dtypes + complex_dtypes
        metafunc.parametrize('rc_dtype', dtypes, ids=[str(x) for x in dtypes])

    if 'dtype_to_broadcast' in metafunc.funcargnames:

        vals = []
        ids = []

        # a simple dtype
        vals.append(numpy.float32)
        ids.append('simple')

        # numpy itemsize == 9, but on device it will be aligned to 4, so the total size will be 12
        dtype = numpy.dtype([('val1', numpy.int32), ('val2', numpy.int32), ('pad', numpy.int8)])
        vals.append(dtype)
        ids.append("small_pad")

        dtype_nested = numpy.dtype([
            ('val1', numpy.int32), ('pad', numpy.int8)])
        dtype = numpy.dtype([
            ('val1', numpy.int32),
            ('val2', numpy.int16),
            ('nested', dtype_nested)])
        vals.append(dtype)
        ids.append("nested")

        dtype_nested = numpy.dtype(dict(
            names=['val1', 'pad'],
            formats=[numpy.int8, numpy.int8]))
        dtype = numpy.dtype(dict(
            names=['pad', 'struct_arr', 'regular_arr'],
            formats=[
                numpy.int32,
                numpy.dtype((dtype_nested, 2)),
                numpy.dtype((numpy.int16, (2, 3)))]))
        vals.append(dtype)
        ids.append("nested_array")

        metafunc.parametrize('dtype_to_broadcast', vals, ids=ids)


def get_test_computation(arr_t):
    return PureParallel(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        """
        <%
            all_idxs = ", ".join(idxs)
        %>
        ${output.store_idx}(${all_idxs}, ${input.load_idx}(${all_idxs}));
        """)


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


in_shapes = [(30,), (20, 1), (10, 20, 1)]
@pytest.mark.parametrize('in_shape', in_shapes, ids=[str(in_shape) for in_shape in in_shapes])
def test_broadcasted_copy(some_thr, in_shape):

    dtype = numpy.int32
    shape = (10, 20, 30)

    input_ = get_test_array(in_shape, dtype)
    input_dev = some_thr.to_device(input_)
    output_dev = some_thr.array(shape, dtype)

    output_ref = numpy.broadcast_to(input_, shape)

    test = get_test_computation(output_dev)
    copy = tr.copy_broadcasted(input_dev, output_dev)

    test.parameter.input.connect(copy, copy.output, input_prime=copy.input)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), output_ref)


def test_cast(some_thr):

    data = get_test_array((1000,), numpy.float32, high=10)
    data_dev = some_thr.to_device(data)

    test = get_test_computation(Type(numpy.int32, (1000,)))
    cast = tr.cast(data, numpy.int32)

    test.parameter.input.connect(cast, cast.output, input_prime=cast.input)
    testc = test.compile(some_thr)

    output_dev = some_thr.empty_like(test.parameter.output)

    testc(output_dev, data_dev)

    assert diff_is_negligible(output_dev.get(), numpy.floor(data).astype(numpy.int32))


def test_add_param(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p1 = get_test_array((1,), any_dtype)[0]
    p2 = get_test_array((1,), any_dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    add = tr.add_param(input_dev, any_dtype)

    test.parameter.input.connect(add, add.output, input_prime=add.input, p1=add.param)
    test.parameter.output.connect(add, add.input, output_prime=add.output, p2=add.param)
    testc = test.compile(some_thr)

    testc(output_dev, p1, input_dev, p2)
    assert diff_is_negligible(output_dev.get(), input + p1 + p2)


def test_add_const(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p1 = get_test_array((1,), any_dtype)[0]
    p2 = get_test_array((1,), any_dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    add1 = tr.add_const(input_dev, p1)
    add2 = tr.add_const(input_dev, p2)

    test.parameter.input.connect(add1, add1.output, input_prime=add1.input)
    test.parameter.output.connect(add2, add2.input, output_prime=add2.output)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), input + p1 + p2)


def test_mul_param(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p1 = get_test_array((1,), any_dtype)[0]
    p2 = get_test_array((1,), any_dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    scale = tr.mul_param(input_dev, any_dtype)

    test.parameter.input.connect(scale, scale.output, input_prime=scale.input, p1=scale.param)
    test.parameter.output.connect(scale, scale.input, output_prime=scale.output, p2=scale.param)
    testc = test.compile(some_thr)

    testc(output_dev, p1, input_dev, p2)
    assert diff_is_negligible(output_dev.get(), input * p1 * p2)


def test_mul_const(some_thr, any_dtype):

    input = get_test_array((1000,), any_dtype)
    p1 = get_test_array((1,), any_dtype)[0]
    p2 = get_test_array((1,), any_dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    scale1 = tr.mul_const(input_dev, p1)
    scale2 = tr.mul_const(input_dev, p2)

    test.parameter.input.connect(scale1, scale1.output, input_prime=scale1.input)
    test.parameter.output.connect(scale2, scale2.input, output_prime=scale2.output)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), input * p1 * p2)


def test_div_param(some_thr):

    dtype = numpy.float32

    input = get_test_array((1000,), dtype)
    p1 = get_test_array((1,), dtype)[0]
    p2 = get_test_array((1,), dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    scale = tr.div_param(input_dev, dtype)

    test.parameter.input.connect(scale, scale.output, input_prime=scale.input, p1=scale.param)
    test.parameter.output.connect(scale, scale.input, output_prime=scale.output, p2=scale.param)
    testc = test.compile(some_thr)

    testc(output_dev, p1, input_dev, p2)
    assert diff_is_negligible(output_dev.get(), input / p1 / p2)


def test_div_const(some_thr):

    dtype = numpy.float32

    input = get_test_array((1000,), dtype)
    p1 = get_test_array((1,), dtype)[0]
    p2 = get_test_array((1,), dtype)[0]
    input_dev = some_thr.to_device(input)
    output_dev = some_thr.empty_like(input_dev)

    test = get_test_computation(input_dev)
    scale1 = tr.div_const(input_dev, p1)
    scale2 = tr.div_const(input_dev, p2)

    test.parameter.input.connect(scale1, scale1.output, input_prime=scale1.input)
    test.parameter.output.connect(scale2, scale2.input, output_prime=scale2.output)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), input / p1 / p2)


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


@pytest.mark.parametrize('order', [1, 2, 0.5])
def test_norm_param(some_thr, rc_dtype, order):

    input_ = get_test_array((1000,), rc_dtype)
    input_dev = some_thr.to_device(input_)

    norm = tr.norm_param(input_dev)

    output_dev = some_thr.empty_like(norm.output)

    test = get_test_computation(output_dev)
    test.parameter.input.connect(norm, norm.output, input_prime=norm.input, order=norm.order)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev, order)
    assert diff_is_negligible(output_dev.get(), numpy.abs(input_) ** order)


@pytest.mark.parametrize('order', [1, 2, 0.5])
def test_norm_const(some_thr, rc_dtype, order):

    input_ = get_test_array((1000,), rc_dtype)
    input_dev = some_thr.to_device(input_)

    norm = tr.norm_const(input_dev, order)

    output_dev = some_thr.empty_like(norm.output)

    test = get_test_computation(output_dev)
    test.parameter.input.connect(norm, norm.output, input_prime=norm.input)
    testc = test.compile(some_thr)

    testc(output_dev, input_dev)
    assert diff_is_negligible(output_dev.get(), numpy.abs(input_) ** order)


def test_broadcast_const(some_thr, dtype_to_broadcast):

    dtype = dtypes.align(dtype_to_broadcast)
    const = get_test_array(1, dtype)[0]

    output_ref = numpy.empty((1000,), dtype)
    output_ref[:] = const

    output_dev = some_thr.empty_like(output_ref)

    test = get_test_computation(output_dev)
    bc = tr.broadcast_const(output_dev, const)
    test.parameter.input.connect(bc, bc.output)
    testc = test.compile(some_thr)

    testc(output_dev)
    assert diff_is_negligible(output_dev.get(), output_ref)


def test_broadcast_param(some_thr, dtype_to_broadcast):

    dtype = dtypes.align(dtype_to_broadcast)
    param = get_test_array(1, dtype)[0]

    output_ref = numpy.empty((1000,), dtype)
    output_ref[:] = param

    output_dev = some_thr.empty_like(output_ref)

    test = get_test_computation(output_dev)
    bc = tr.broadcast_param(output_dev)
    test.parameter.input.connect(bc, bc.output, param=bc.param)
    testc = test.compile(some_thr)

    testc(output_dev, param)
    assert diff_is_negligible(output_dev.get(), output_ref)
