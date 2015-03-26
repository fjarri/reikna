import itertools
import time

import numpy
import pytest

from helpers import *

from reikna.helpers import product
from reikna.fftshift import FFTShift
import reikna.cluda.dtypes as dtypes


def check_errors(thr, shape_and_axes):

    dtype = numpy.complex64

    shape, axes = shape_and_axes

    data = get_test_array(shape, dtype)

    shift = FFTShift(data, axes=axes)
    shiftc = shift.compile(thr)

    data_dev = thr.to_device(data)
    shiftc(data_dev, data_dev)
    data_ref = numpy.fft.fftshift(data, axes=axes)
    assert diff_is_negligible(data_dev.get(), data_ref)


@pytest.mark.parametrize('shape_and_axes', [((512,), (0,))])
def test_1d(thr, shape_and_axes):
    check_errors(thr, shape_and_axes)
