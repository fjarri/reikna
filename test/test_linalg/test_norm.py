import numpy
import pytest

from helpers import *
from reikna.linalg import EntrywiseNorm


def reference_norm(arr, order=1, axes=None):
    if axes is None:
        axes = list(range(len(arr.shape)))

    arr = numpy.abs(arr)
    out_dtype = arr.dtype

    arr = arr ** order
    for axis in reversed(sorted(axes)):
        arr = arr.sum(axis)

    # explicit cast to preven numpy promoting arr to float64 from float32 if it is 0-dimensional
    res = arr ** numpy.cast[out_dtype](1. / order)
    return res


def check_norm(thr, shape, dtype, order, axes):
    a = get_test_array(shape, dtype)
    a_dev = thr.to_device(a)

    norm = EntrywiseNorm(a_dev, order=order, axes=axes)

    b_dev = thr.empty_like(norm.parameter.output)
    b_ref = reference_norm(a, order=order, axes=axes)

    normc = norm.compile(thr)
    normc(b_dev, a_dev)

    assert diff_is_negligible(b_dev.get(), b_ref)


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('order', [0.5, 1, 2])
def test_all_axes(thr, dtype, order):
    check_norm(thr, 100, dtype, order, None)


axes_vals = [None, (0,), (2,), (1, 2)]
axes_ids = [str(axes) for axes in axes_vals]
@pytest.mark.parametrize('axes', axes_vals, ids=axes_ids)
def test_some_axes(thr, axes):
    check_norm(thr, (20, 10, 5), numpy.float32, 2, axes)
