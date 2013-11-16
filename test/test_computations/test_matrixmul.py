import time
import numpy
import pytest

from helpers import *

from reikna.matrixmul import MatrixMul
import reikna.cluda.dtypes as dtypes
from reikna.cluda import OutOfResourcesError
from reikna.helpers import product


def pytest_generate_tests(metafunc):

    if 'perf_bwo' in metafunc.funcargnames:
        bwos = [8, 16, 32]
        ids=["8x8", "16x16", "32x32"]
        metafunc.parametrize('perf_bwo', bwos, ids=ids)

    if 'perf_shapes' in metafunc.funcargnames:

        shapes = []
        for s in [(64, 64), (256, 256), (1024, 1024)]:
            shapes.append((s, s))

        for s in [(1024, 16, 16), (256, 32, 32), (512, 25, 25)]:
            shapes.append((s, s[1:]))
            shapes.append((s, s))

        ids = []
        for s in shapes:
            ids.append(str(s[0]) + 'x' + str(s[1]))

        metafunc.parametrize('perf_shapes', shapes, ids=ids)

    if 'shapes' in metafunc.funcargnames:

        shapes = [
            ((15, 17), (17, 3)),
            ((122, 5), (5, 1000)),
            ((45, 99), (30, 99, 40)),
            ((12, 56, 78), (12, 78, 44))
        ]

        ids = [str(s1) + 'x' + str(s2) for s1, s2 in shapes]

        metafunc.parametrize('shapes', shapes, ids=ids)

    if 'arg_dtypes' in metafunc.funcargnames:

        arg_dtypes = [(False, False), (False, True), (True, False), (True, True)]
        mark = lambda x: 'c' if x else 'r'
        ids = [mark(t1) + mark(t2) for t1, t2 in arg_dtypes]
        metafunc.parametrize('arg_dtypes', arg_dtypes, ids=ids)


def ref_dot(a, b):
    a_batch = product(a.shape[:-2])
    b_batch = product(b.shape[:-2])

    assert a_batch == b_batch or a_batch == 1 or b_batch == 1

    a = a.reshape(a_batch, a.shape[-2], a.shape[-1])
    b = b.reshape(b_batch, b.shape[-2], b.shape[-1])

    out_batch = max(a_batch, b_batch)
    out_shape = (out_batch, a.shape[-2], b.shape[-1])
    out_dtype = numpy.result_type(a.dtype, b.dtype)
    out = numpy.empty(out_shape, out_dtype)

    for i in range(out_batch):
        ai = 0 if a_batch == 1 else i
        bi = 0 if b_batch == 1 else i
        out[i] = numpy.dot(a[ai], b[bi])

    if a_batch == b_batch == 1:
        out = out.reshape(out.shape[-2], out_shape[-1])

    return out


def check_errors(thr, a_shape, a_dtype, b_shape, b_dtype, transposed_a=False, transposed_b=False):
    a = get_test_array(a_shape, a_dtype)
    b = get_test_array(b_shape, b_dtype)

    a_ref = a.T if transposed_a else a
    b_ref = b.T if transposed_b else b

    res_ref = ref_dot(a_ref, b_ref)

    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    res_dev = thr.empty_like(res_ref)

    dot = MatrixMul(a_dev, b_dev, out_arr=res_dev,
        transposed_a=transposed_a, transposed_b=transposed_b)
    dotc = dot.compile(thr)
    dotc(res_dev, a_dev, b_dev)

    assert diff_is_negligible(res_dev.get(), res_ref)


def test_errors(thr_and_double, shapes, arg_dtypes):

    thr, double = thr_and_double
    s1, s2 = shapes
    c1, c2 = arg_dtypes

    dtype = numpy.float64 if double else numpy.float32
    dtype1 = dtypes.complex_for(dtype) if c1 else dtype
    dtype2 = dtypes.complex_for(dtype) if c2 else dtype

    check_errors(thr, s1, dtype1, s2, dtype2)


transposed_sizes = [(100, 200, 300), (3, 4, 5), (64, 23, 79)]
@pytest.mark.parametrize('transposed_size',
    transposed_sizes, ids=[str(x) for x in transposed_sizes])
@pytest.mark.parametrize('transposed_a', [False, True], ids=['A.T', 'A'])
@pytest.mark.parametrize('transposed_b', [False, True], ids=['B.T', 'B'])
def test_transposed(thr, transposed_size, transposed_a, transposed_b):
    a_size, convolution_size, b_size = transposed_size

    if transposed_a:
        a_shape = (convolution_size, a_size)
    else:
        a_shape = (a_size, convolution_size)

    if transposed_b:
        b_shape = (b_size, convolution_size)
    else:
        b_shape = (convolution_size, b_size)

    check_errors(thr, a_shape, numpy.float32, b_shape, numpy.float32,
        transposed_a=transposed_a, transposed_b=transposed_b)


def check_performance(thr_and_double, shape1, shape2, bwo):

    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32

    a = get_test_array(shape1, dtype)
    b = get_test_array(shape2, dtype)

    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    res_ref = ref_dot(a, b)
    res_dev = thr.array(res_ref.shape, dtype=dtype)

    dot = MatrixMul(a_dev, b_dev, out_arr=res_dev, block_width_override=bwo)

    try:
        dotc = dot.compile(thr)
    except ValueError:
        pytest.skip()

    attempts = 10
    t1 = time.time()
    for i in range(attempts):
        dotc(res_dev, a_dev, b_dev)
    thr.synchronize()
    t2 = time.time()

    assert diff_is_negligible(thr.from_device(res_dev), res_ref)

    return (t2 - t1) / attempts, product(res_ref.shape) * shape1[-1] * 2


@pytest.mark.perf
@pytest.mark.returns('GFLOPS')
def test_performance_shape(thr_and_double, perf_shapes):
    return check_performance(thr_and_double, perf_shapes[0], perf_shapes[1], None)


@pytest.mark.perf
@pytest.mark.returns('GFLOPS')
def test_performance_block_width(thr_and_double, perf_bwo):
    return check_performance(thr_and_double, (512, 512), (512, 512), perf_bwo)
