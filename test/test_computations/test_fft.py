import itertools

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


def test_errors(ctx_and_double, shape_and_axes):

    ctx, double = ctx_and_double
    dtype = numpy.complex128 if double else numpy.complex64

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
