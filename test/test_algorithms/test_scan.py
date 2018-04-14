import time

import numpy
import pytest

from reikna.algorithms import Scan
import reikna.helpers as helpers
import reikna.cluda.dtypes as dtypes

from helpers import *


perf_shapes = [(1024 * 1024,), (1024 * 1024 * 8,), (1024 * 1024 * 64,)]
@pytest.fixture(params=perf_shapes, ids=list(map(str, perf_shapes)))
def large_perf_shape(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["exclusive", "inclusive"])
def exclusive(request):
    return request.param


def ref_scan(arr, axes=None, exclusive=False):
    if axes is None:
        res = numpy.cumsum(arr).reshape(arr.shape)
    else:
        axes = helpers.normalize_axes(arr.ndim, axes)
        transpose_to, transpose_from = helpers.make_axes_innermost(arr.ndim, axes)
        unchanged_ndim = arr.ndim - len(axes)
        temp = arr.transpose(transpose_to)
        temp2 = temp.reshape(
            temp.shape[:unchanged_ndim] + (helpers.product(temp.shape[unchanged_ndim:]),))
        temp2 = numpy.cumsum(temp2, axis=-1)
        res = temp2.reshape(temp.shape).transpose(transpose_from)

    if exclusive:
        res -= arr

    return res


def check_scan(
        thr, shape, axes, exclusive=False,
        measure_time=False, dtype=numpy.int64, max_work_group_size=None):

    arr = get_test_array(shape, dtype)

    scan = Scan(
        arr, axes=axes, exclusive=exclusive,
        max_work_group_size=max_work_group_size).compile(thr)

    arr_dev = thr.to_device(arr)
    res_dev = thr.to_device(numpy.ones_like(arr) * (-1))#thr.empty_like(arr)

    if measure_time:
        attempts = 10
        times = []
        for i in range(attempts):
            t1 = time.time()
            scan(res_dev, arr_dev)
            thr.synchronize()
            times.append(time.time() - t1)
        min_time = min(times)
    else:
        scan(res_dev, arr_dev)
        min_time = None

    res_test = res_dev.get()

    res_ref = ref_scan(arr, axes=axes, exclusive=exclusive)

    assert diff_is_negligible(res_ref, res_test)

    return min_time


def test_scan_correctness(thr, corr_shape, exclusive):
    check_scan(thr, corr_shape, axes=None, exclusive=exclusive, max_work_group_size=512)


def test_scan_multiple_axes(thr):
    check_scan(thr, (10, 20, 30, 40), axes=(2,3))


def test_scan_non_innermost_axes(thr):
    check_scan(thr, (10, 20, 30, 40), axes=(1,2))


@pytest.mark.perf
@pytest.mark.returns('GB/s')
def test_large_scan_performance(thr, large_perf_shape, exclusive):
    """
    Large problem sizes.
    """
    dtype = dtypes.normalize_type(numpy.int64)
    min_time = check_scan(
        thr, large_perf_shape, dtype=dtype, axes=None, exclusive=exclusive, measure_time=True)
    return min_time, helpers.product(large_perf_shape) * dtype.itemsize
