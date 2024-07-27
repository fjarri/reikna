import numpy
import pytest
from grunnur import Array

from helpers import *
from reikna.linalg import EntrywiseNorm


def reference_norm(arr, order=1, axes=None):
    if axes is None:
        axes = list(range(len(arr.shape)))

    arr = numpy.abs(arr)
    out_dtype = arr.dtype

    arr = arr**order
    for axis in axes:
        arr = arr.sum(axis, keepdims=True)

    # explicit cast to preven numpy promoting arr to float64 from float32 if it is 0-dimensional
    res = arr ** numpy.asarray(1.0 / order, out_dtype)
    return res


def check_norm(queue, shape, dtype, order, axes):
    a = get_test_array(shape, dtype)
    a_dev = Array.from_host(queue, a)

    norm = EntrywiseNorm(a_dev, order=order, axes=axes)

    b_dev = Array.empty_like(queue.device, norm.parameter.output)
    b_ref = reference_norm(a, order=order, axes=axes)

    normc = norm.compile(queue.device)
    normc(queue, b_dev, a_dev)

    assert diff_is_negligible(b_dev.get(queue), b_ref)


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.complex64], ids=["float32", "complex64"])
@pytest.mark.parametrize("order", [0.5, 1, 2])
def test_all_axes(queue, dtype, order):
    check_norm(queue, 100, dtype, order, None)


axes_vals = [None, (0,), (2,), (1, 2)]
axes_ids = [str(axes) for axes in axes_vals]


@pytest.mark.parametrize("axes", axes_vals, ids=axes_ids)
def test_some_axes(queue, axes):
    check_norm(queue, (20, 10, 5), numpy.float32, 2, axes)
