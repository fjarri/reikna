import time

import numpy
import pytest

from reikna.algorithms import Scan, Predicate, predicate_sum
from reikna.cluda import Snippet
import reikna.helpers as helpers
import reikna.cluda.dtypes as dtypes

from helpers import *


perf_shapes = [(1024 * 1024,), (1024 * 1024 * 8,), (1024 * 1024 * 64,)]
@pytest.fixture(params=perf_shapes, ids=list(map(str, perf_shapes)))
def large_perf_shape(request):
    return request.param


corr_shapes = [(15,), (511,), (512,), (513,), (512*512+1,), (512*512*4+5,)]
@pytest.fixture(params=corr_shapes, ids=list(map(str, corr_shapes)))
def corr_shape(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["exclusive", "inclusive"])
def exclusive(request):
    return request.param


@pytest.fixture(params=[1, 2, 4, 8])
def seq_size(request):
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
        measure_time=False, dtype=numpy.int64, max_work_group_size=None, predicate=None,
        seq_size=None):

    # Note: the comparison will only work if the custom predicate is
    # functionally equivalent to `predicate_sum`.
    if predicate is None:
        predicate = predicate_sum(dtype)

    arr = get_test_array(shape, dtype)

    scan = Scan(
        arr, predicate, axes=axes, exclusive=exclusive,
        max_work_group_size=max_work_group_size, seq_size=seq_size).compile(thr)

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


def test_scan_custom_predicate(thr):
    predicate = Predicate(
        Snippet.create(lambda v1, v2: "return ${v1} + ${v2};"),
        0)
    check_scan(thr, (10, 20, 30, 40), axes=(1,2), predicate=predicate)


def test_scan_structure_type(thr, exclusive):

    shape = (100, 100)
    dtype = dtypes.align(numpy.dtype([
        ('i1', numpy.uint32),
        ('nested', numpy.dtype([
            ('v', numpy.uint64),
            ])),
        ('i2', numpy.uint32)
        ]))

    a = get_test_array(shape, dtype)
    a_dev = thr.to_device(a)

    # Have to construct the resulting array manually,
    # since numpy cannot scan arrays with struct dtypes.
    b_ref = numpy.empty(shape, dtype)
    b_ref['i1'] = ref_scan(a['i1'], axes=0, exclusive=exclusive)
    b_ref['nested']['v'] = ref_scan(a['nested']['v'], axes=0, exclusive=exclusive)
    b_ref['i2'] = ref_scan(a['i2'], axes=0, exclusive=exclusive)

    predicate = Predicate(
        Snippet.create(lambda v1, v2: """
            ${ctype} result = ${v1};
            result.i1 += ${v2}.i1;
            result.nested.v += ${v2}.nested.v;
            result.i2 += ${v2}.i2;
            return result;
            """,
            render_kwds=dict(
                ctype=dtypes.ctype_module(dtype))),
        numpy.zeros(1, dtype)[0])

    scan = Scan(a_dev, predicate, axes=(0,), exclusive=exclusive)

    b_dev = thr.empty_like(scan.parameter.output)

    scanc = scan.compile(thr)
    scanc(b_dev, a_dev)
    b_res = b_dev.get()

    assert diff_is_negligible(b_res, b_ref)


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


@pytest.mark.perf
@pytest.mark.returns('GB/s')
def test_small_scan_performance(thr, exclusive, seq_size):
    """
    Small problem sizes, big batches.
    """
    dtype = dtypes.normalize_type(numpy.complex128)
    shape = (500, 2, 2, 512)
    min_time = check_scan(
        thr, shape, dtype=dtype, axes=(-1,), exclusive=exclusive,
        measure_time=True, seq_size=seq_size)
    return min_time, helpers.product(shape) * dtype.itemsize
