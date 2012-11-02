import itertools
import time

import numpy
import pytest

from helpers import *

from tigger.helpers import product
from tigger.fft import FFT
import tigger.cluda.dtypes as dtypes


def pytest_generate_tests(metafunc):
    if 'shape_and_axes' in metafunc.funcargnames:
        shapes = []

        for x in [3, 8, 9, 10, 11, 13, 20]:
            shapes.append((2 ** x,))

        for x, y in itertools.product([4, 7, 8, 10], [4, 7, 8, 10]):
            shapes.append((2 ** x, 2 ** y))

        for x, y, z in itertools.product([4, 7, 10], [4, 7, 10], [4, 7, 10]):
            shapes.append((2 ** x, 2 ** y, 2 ** z))

        batch_sizes = [1, 16, 128, 1024, 4096]

        mem_limit = 2 ** 20

        vals = []
        ids = []
        for shape, batch in itertools.product(shapes, batch_sizes):
            if product(shape) * batch <= mem_limit:
                if batch == 1:
                    vals.append((shape, None))
                else:
                    vals.append(((batch,) + shape, tuple(range(1, len(shape) + 1))))
                ids.append(str(batch) + "x" + str(shape))

        metafunc.parametrize('shape_and_axes', vals, ids=ids)

    elif 'non2batch_shape_and_axes' in metafunc.funcargnames:
        def idgen(shape_and_axes):
            shape, axes = shape_and_axes
            assert len(axes) == 1
            outer_batch = shape[:axes[0]]
            inner_batch = shape[axes[0]+1:]
            return ((str(outer_batch) + "x") if len(outer_batch) > 0 else "") + \
                str(shape[axes[0]]) + "x" + str(inner_batch)

        vals = [
            ((17, 16), (1,)),
            ((177, 256), (1,)),
            ((39, 16, 7), (1,)),
            ((17, 16, 131), (1,)),
            ((7, 1024, 11), (1,)),
            ((5, 1024, 57), (1,))]

        metafunc.parametrize('non2batch_shape_and_axes', vals, ids=map(idgen, vals))

    elif 'non2problem_shape_and_axes' in metafunc.funcargnames:

        def idgen(non2problem_shape_and_axes):
            shape, axes = non2problem_shape_and_axes
            return str(shape) + 'over' + str(axes)

        vals = [
            ((17, 15), (1,)),
            ((17, 17), (1,)),
            ((19, 4095), (1,)),
            ((19, 4097), (1,)),
            ((39, 31, 7), (1,)),
            ((39, 33, 7), (1,)),
            ((3, 255, 7), (1,)),
            ((3, 257, 7), (1,)),
            ((17, 200, 131), (0, 1)),
            ((7, 1000, 11), (1, 2)),
            ((15, 900, 57), (0, 1, 2))]

        metafunc.parametrize('non2problem_shape_and_axes', vals, ids=map(idgen, vals))

    elif 'perf_shape_and_axes' in metafunc.funcargnames:
        log_shapes = [
            (4,), (10,), (13,), # 1D
            (4, 4), (7, 7), (10, 10), # 2D
            (3, 3, 6), (4, 4, 4), (4, 4, 7), (5, 5, 7), (7, 7, 7) # 3D
        ]

        mem_limit = 4 * 2**20
        vals = []
        ids = []
        for log_shape in log_shapes:
            shape = tuple(2 ** x for x in log_shape)
            batch = mem_limit / product(shape)
            vals.append(((batch,) + shape, tuple(range(1, len(shape) + 1))))
            ids.append(str(batch) + "x" + str(shape))

        metafunc.parametrize('perf_shape_and_axes', vals, ids=ids)


def test_errors(ctx, shape_and_axes):

    dtype = numpy.complex64

    shape, axes = shape_and_axes

    data = get_test_array(shape, dtype)
    data_dev = ctx.to_device(data)
    res_dev = ctx.empty_like(data_dev)

    fft = FFT(ctx).prepare_for(res_dev, data_dev, None, axes=axes)

    # forward transform
    fft(res_dev, data_dev, -1)
    fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(res_dev.get(), fwd_ref)

    # inverse transform
    fft(res_dev, data_dev, 1)
    inv_ref = numpy.fft.ifftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(res_dev.get(), inv_ref)


def test_non2problem(ctx, non2problem_shape_and_axes):

    dtype = numpy.complex64

    shape, axes = non2problem_shape_and_axes

    data = get_test_array(shape, dtype)
    data_dev = ctx.to_device(data)
    res_dev = ctx.empty_like(data_dev)

    fft = FFT(ctx).prepare_for(res_dev, data_dev, None, axes=axes)

    # forward transform
    fft(res_dev, data_dev, -1)
    fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(res_dev.get(), fwd_ref)

    # inverse transform
    fft(res_dev, data_dev, 1)
    inv_ref = numpy.fft.ifftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(res_dev.get(), inv_ref)


def test_non2batch(ctx, non2batch_shape_and_axes):
    """
    Tests that the normal algoritms supports both inner and outer batches that are not powers of 2.
    Batches here are those part of ``shape`` that are not referenced in ``axes``.
    """

    dtype = numpy.complex64

    shape, axes = non2batch_shape_and_axes

    data = get_test_array(shape, dtype)
    data_dev = ctx.to_device(data)
    res_dev = ctx.empty_like(data_dev)

    fft = FFT(ctx).prepare_for(res_dev, data_dev, None, axes=axes)

    # forward transform
    fft(res_dev, data_dev, -1)
    fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(res_dev.get(), fwd_ref)


@pytest.mark.perf
@pytest.mark.returns('GFLOPS')
def test_power_of_2_performance(ctx_and_double, perf_shape_and_axes):
    ctx, double = ctx_and_double

    shape, axes = perf_shape_and_axes
    dtype = numpy.complex128 if double else numpy.complex64

    data = get_test_array(shape, dtype)
    data_dev = ctx.to_device(data)
    res_dev = ctx.empty_like(data_dev)

    fft = FFT(ctx).prepare_for(res_dev, data_dev, None, axes=axes)

    attempts = 10
    t1 = time.time()
    for i in range(attempts):
        fft(res_dev, data_dev, -1)
    ctx.synchronize()
    t2 = time.time()
    dev_time = (t2 - t1) / attempts

    fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(res_dev.get(), fwd_ref)

    return dev_time, product(shape) * sum([numpy.log2(shape[a]) for a in axes]) * 5
