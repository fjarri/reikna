import itertools
import numpy
import pytest

from helpers import *

from reikna.helpers import product
from reikna.dht import DHT, harmonic, get_spatial_grid
import reikna.cluda.dtypes as dtypes


class TestFunction:

    def __init__(self, mshape, dtype, order=1, batch=1, mdata=None):
        self.order = order
        self.mshape = tuple(mshape)
        self.full_mshape = (batch,) + self.mshape
        self.batch = batch
        self.dtype = dtype

        if mdata is None:
            self.mdata = get_test_array(self.full_mshape, self.dtype)
        else:
            self.mdata = numpy.array(mdata).astype(self.dtype)

        self.harmonics = [harmonic(n) for n in range(max(self.mshape))]

    def __call__(self, *xs):
        if len(xs) > 1:
            xxs = numpy.meshgrid(*xs, indexing="ij")
        else:
            xxs = xs
        res = numpy.zeros((self.batch,) + xxs[0].shape, self.dtype)

        for b in range(self.batch):
            for modenums in itertools.product(*[range(modes) for modes in self.mshape]):
                coords = (b,) + modenums
                coeff = self.mdata[coords]
                res[b] += coeff * \
                    product([self.harmonics[modenum](xx) for modenum, xx in zip(modenums, xxs)])

        return res ** self.order


def pytest_generate_tests(metafunc):

    if 'fo_shape' in metafunc.funcargnames:
        vals = [(5,), (20,), (3, 7), (10, 11), (5, 6, 7), (10, 11, 12)]
        metafunc.parametrize('fo_shape', vals, ids=list(map(str, vals)))

    if 'fo_batch' in metafunc.funcargnames:
        vals = [1, 10]
        metafunc.parametrize('fo_batch', vals, ids=list(map(str, vals)))

    if 'fo_add_points' in metafunc.funcargnames:
        vals = ['0', '1', '1,2,...']
        metafunc.parametrize('fo_add_points', vals)


def check_errors_first_order(ctx, mshape, batch, add_points=None, dtype=numpy.complex64):

    test_func = TestFunction(mshape, dtype, batch=batch, order=1)

    if add_points is None:
        add_points = [0] * len(mshape)
    xs = [get_spatial_grid(n, 1, add_points=ap)[0] for n, ap in zip(mshape, add_points)]

    xdata = test_func(*xs)
    xdata_dev = ctx.to_device(xdata)
    mdata_dev = ctx.array((batch,) + mshape, dtype)
    axes = range(1, len(mshape)+1)

    dht_fw = DHT(ctx).prepare_for(mdata_dev, xdata_dev, inverse=False, axes=axes)
    dht_inv = DHT(ctx).prepare_for(xdata_dev, mdata_dev, inverse=True, axes=axes)

    # forward transform
    dht_fw(mdata_dev, xdata_dev)
    assert diff_is_negligible(mdata_dev.get(), test_func.mdata)

    # inverse transform
    dht_inv(xdata_dev, mdata_dev)
    assert diff_is_negligible(xdata_dev.get(), xdata)

def test_first_order_errors(ctx, fo_shape, fo_batch, fo_add_points):
    if fo_add_points == '0':
        add_points = None
    elif fo_add_points == '1':
        add_points = [1] * len(fo_shape)
    else:
        add_points = range(1, len(fo_shape) + 1)

    check_errors_first_order(ctx, fo_shape, fo_batch,
        add_points=add_points, dtype=numpy.complex64)
