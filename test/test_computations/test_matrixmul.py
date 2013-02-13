import time
import numpy
import pytest

from helpers import *

from tigger.matrixmul import MatrixMul
import tigger.cluda.dtypes as dtypes
from tigger.cluda import OutOfResourcesError
from tigger.helpers import product


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


def test_errors(ctx_and_double, shapes, arg_dtypes):

    ctx, double = ctx_and_double
    s1, s2 = shapes
    c1, c2 = arg_dtypes

    dtype = numpy.float64 if double else numpy.float32
    dtype1 = dtypes.complex_for(dtype) if c1 else dtype
    dtype2 = dtypes.complex_for(dtype) if c2 else dtype

    a = get_test_array(s1, dtype1)
    b = get_test_array(s2, dtype2)
    res_ref = ref_dot(a, b)

    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    res_dev = ctx.empty_like(res_ref)

    dot = MatrixMul(ctx).prepare_for(res_dev, a_dev, b_dev)
    dot(res_dev, a_dev, b_dev)

    assert diff_is_negligible(ctx.from_device(res_dev), res_ref)


def check_performance(ctx_and_double, shape1, shape2, bwo):

    ctx, double = ctx_and_double
    dtype = numpy.float64 if double else numpy.float32

    a = get_test_array(shape1, dtype)
    b = get_test_array(shape2, dtype)

    a_dev = ctx.to_device(a)
    b_dev = ctx.to_device(b)
    res_ref = ref_dot(a, b)
    res_dev = ctx.array(res_ref.shape, dtype=dtype)

    try:
        dot = MatrixMul(ctx).prepare_for(res_dev, a_dev, b_dev, block_width_override=bwo)
    except ValueError:
        pytest.skip()

    attempts = 10
    t1 = time.time()
    for i in range(attempts):
        dot(res_dev, a_dev, b_dev)
    ctx.synchronize()
    t2 = time.time()

    assert diff_is_negligible(ctx.from_device(res_dev), res_ref)

    return (t2 - t1) / attempts, product(res_ref.shape) * shape1[-1] * 2


@pytest.mark.perf
@pytest.mark.returns('GFLOPS')
def test_performance_shape(ctx_and_double, perf_shapes):
    return check_performance(ctx_and_double, perf_shapes[0], perf_shapes[1], None)


@pytest.mark.perf
@pytest.mark.returns('GFLOPS')
def test_performance_block_width(ctx_and_double, perf_bwo):
    return check_performance(ctx_and_double, (512, 512), (512, 512), perf_bwo)
