import itertools
import numpy
import pytest

from helpers import *

from reikna.helpers import product
from reikna.dht import DHT, harmonic, get_spatial_grid
import reikna.cluda.dtypes as dtypes


class FunctionHelper:
    """
    Encapsulates creation of functions in mode and coordinate space used for DHT tests.
    """

    def __init__(self, mshape, dtype, order=1, batch=None, modes=None):
        self.order = order
        self.mshape = tuple(mshape)
        self.full_mshape = ((batch,) if batch is not None else tuple()) + self.mshape
        self.batch = batch
        self.dtype = dtype

        if modes is None:
            self.modes = FunctionHelper.generate_modes(mshape, dtype, batch=batch)
        else:
            self.modes = list(modes)

        self.mdata = numpy.zeros(self.full_mshape, self.dtype)
        for coeff, coord in self.modes:
            self.mdata[coord] = coeff

        self.harmonics = [harmonic(n) for n in range(max(self.mshape))]

    @staticmethod
    def generate_modes(mshape, dtype, batch=None, random=True):
        """
        Generates list of sparse modes for the problem of given shape.
        """

        max_modes_per_batch = 20

        modelist = []
        if product(mshape) <= max_modes_per_batch:
            # If there are not many modes, fill all of them
            modenums = itertools.product(*[range(modes) for modes in mshape])
            if batch is not None:
                for b in range(batch):
                    modelist += [((b,) + modenum) for modenum in modenums]
            else:
                modelist += list(modenums)
        else:
            # If there are many modes, fill some random ones
            rand_coord = lambda: tuple(
                numpy.random.randint(0, mshape[i]) for i in range(len(mshape)))

            if batch is not None:
                for b in range(batch):
                    for i in range(max_modes_per_batch):
                        modelist.append((b,) + rand_coord())
            else:
                for i in range(max_modes_per_batch):
                    modelist.append(rand_coord())

        # add corner modes, to make sure extreme cases are still processed correctly
        corner_modes = itertools.product(*[(0, mshape[i]-1) for i in range(len(mshape))])
        for modenum in corner_modes:
            if batch is not None:
                for b in range(batch):
                    modelist.append((b,) + modenum)
            else:
                modelist.append(modenum)

        modelist = set(modelist) # remove duplicates

        # Assign coefficients
        modes = []
        for coord in modelist:
            get_coeff = lambda: numpy.random.normal() if random else 1
            if dtypes.is_complex(dtype):
                coeff = get_coeff() + 1j * get_coeff()
            else:
                coeff = get_coeff()
            coeff = dtype(coeff)

            # scaling coefficients for higher modes because of the lower precision in this case
            modenums = coord if batch is None else coord[1:]
            coeff /= sum(modenums) + 1
            modes.append((coeff, coord))

        return modes

    def __call__(self, *xs):
        """
        Evaluate function in coordinate space for given grid.
        """

        if len(xs) > 1:
            xxs = numpy.meshgrid(*xs, indexing="ij")
        else:
            xxs = xs

        res_shape = ((self.batch,) if self.batch is not None else tuple()) + xxs[0].shape
        res = numpy.zeros(res_shape, self.dtype)

        for coeff, coord in self.modes:
            if self.batch is not None:
                b = coord[0]
                coord = coord[1:]
                target = res[b]
            else:
                target = res

            target += coeff * product([self.harmonics[m](xx) for m, xx in zip(coord, xxs)])

        return res ** self.order


def check_errors_first_order(thr, mshape, batch, add_points=None, dtype=numpy.complex64):

    test_func = FunctionHelper(mshape, dtype, batch=batch, order=1)

    if add_points is None:
        add_points = [0] * len(mshape)
    xs = [get_spatial_grid(n, 1, add_points=ap) for n, ap in zip(mshape, add_points)]

    mdata_dev = thr.array((batch,) + mshape, dtype)
    axes = list(range(1, len(mshape)+1))

    dht_fw = DHT(mdata_dev, inverse=False, axes=axes, add_points=[0] + add_points)
    dht_inv = DHT(mdata_dev, inverse=True, axes=axes, add_points=[0] + add_points)
    dht_fw_c = dht_fw.compile(thr)
    dht_inv_c = dht_inv.compile(thr)

    xdata = test_func(*xs)
    xdata_dev = thr.to_device(xdata)

    # forward transform
    dht_fw_c(mdata_dev, xdata_dev)
    assert diff_is_negligible(mdata_dev.get(), test_func.mdata, atol=1e-6)

    # inverse transform
    dht_inv_c(xdata_dev, mdata_dev)
    assert diff_is_negligible(xdata_dev.get(), xdata, atol=1e-6)


fo_shape_vals = [(5,), (20,), (50,), (3, 7), (10, 11), (5, 6, 7), (10, 11, 12)]
@pytest.mark.parametrize('fo_shape', fo_shape_vals, ids=list(map(str, fo_shape_vals)))
@pytest.mark.parametrize('fo_batch', [1, 10])
@pytest.mark.parametrize('fo_add_points', ['0', '1', '1,2,...'])
def test_first_order_errors(thr, fo_shape, fo_batch, fo_add_points):
    """
    Checks that after the transformation of the manually constructed function in coordinate space
    we get exactly mode numbers used for its construction.
    Also checks that inverse transform returns the initial array.
    """

    if fo_add_points == '0':
        add_points = None
    elif fo_add_points == '1':
        add_points = [1] * len(fo_shape)
    else:
        add_points = list(range(1, len(fo_shape) + 1))

    check_errors_first_order(thr, fo_shape, fo_batch,
        add_points=add_points, dtype=numpy.complex64)


@pytest.mark.parametrize('ho_order', [2, 3])
@pytest.mark.parametrize('ho_shape', [20, 30, 50])
def test_high_order_forward(thr, ho_order, ho_shape):
    """
    Checks that if we change the mode space while keeping mode population the same,
    the result of forward transformation for orders higher than 1 do not change.
    """

    dtype = numpy.float32

    modes = FunctionHelper.generate_modes((ho_shape,), dtype)
    f1 = FunctionHelper((ho_shape,), dtype, order=ho_order, modes=modes)
    f2 = FunctionHelper((ho_shape + 1,), dtype, order=ho_order, modes=modes)

    xs1 = get_spatial_grid(ho_shape, ho_order)
    xs2 = get_spatial_grid(ho_shape + 1, ho_order)

    xdata1 = f1(xs1)
    xdata2 = f2(xs2)

    xdata1_dev = thr.to_device(xdata1)
    xdata2_dev = thr.to_device(xdata2)

    mdata1_dev = thr.array(ho_shape, dtype)
    mdata2_dev = thr.array(ho_shape + 1, dtype)

    dht_fw1 = DHT(mdata1_dev, inverse=False, order=ho_order)
    dht_fw2 = DHT(mdata2_dev, inverse=False, order=ho_order)
    dht_fw1_c = dht_fw1.compile(thr)
    dht_fw2_c = dht_fw2.compile(thr)

    dht_fw1_c(mdata1_dev, xdata1_dev)
    dht_fw2_c(mdata2_dev, xdata2_dev)

    mdata1 = mdata1_dev.get()
    mdata2 = mdata2_dev.get()

    assert diff_is_negligible(mdata1, mdata2[:-1], atol=1e-6)
