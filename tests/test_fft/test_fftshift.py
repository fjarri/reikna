import itertools
import time

import numpy
import pytest
from grunnur import Array, dtypes

from helpers import *
from reikna.fft import FFTShift
from reikna.helpers import product
from reikna.transformations import mul_param


def pytest_generate_tests(metafunc):
    errors_shapes_and_axes = [
        ((10,), (0,)),
        ((11,), (0,)),
        ((9000,), (0,)),
        ((9001,), (0,)),
        ((128, 60), (0, 1)),
        ((127, 60), (0, 1)),
        ((127, 61), (0, 1)),
        ((100, 80, 60), (0, 1, 2)),
        ((101, 80, 61), (0, 1, 2)),
        ((101, 80, 61), (0, 2)),
        ((20, 31, 80, 61), (0, 2)),
    ]

    perf_shapes = [
        (2**4,),  # 1D, small size
        (2**18,),  # 1D, large size
        (2**4, 2**4),  # 2D, small size
        (2**9, 2**9),  # 2D, large size
    ]
    perf_even_shapes_and_axes = []
    perf_odd_shapes_and_axes = []

    mem_limit = 2**22

    for contigous in (True, False):
        for shape in perf_shapes:
            batch = mem_limit // product(shape)
            if contigous:
                full_shape = (batch,) + shape
                axes = tuple(range(1, len(shape) + 1))
            else:
                full_shape = shape + (batch,)
                axes = tuple(range(0, len(shape)))

            perf_even_shapes_and_axes.append((full_shape, axes))

            full_shape = list(full_shape)
            for axis in axes:
                full_shape[axis] -= 1
            perf_odd_shapes_and_axes.append((tuple(full_shape), axes))

    idgen = lambda pair: str(pair[0]) + "_over_" + str(pair[1])

    if "errors_shape_and_axes" in metafunc.fixturenames:
        metafunc.parametrize(
            "errors_shape_and_axes",
            errors_shapes_and_axes,
            ids=list(map(idgen, errors_shapes_and_axes)),
        )

    elif "perf_even_shape_and_axes" in metafunc.fixturenames:
        metafunc.parametrize(
            "perf_even_shape_and_axes",
            perf_even_shapes_and_axes,
            ids=list(map(idgen, perf_even_shapes_and_axes)),
        )

    elif "perf_odd_shape_and_axes" in metafunc.fixturenames:
        metafunc.parametrize(
            "perf_odd_shape_and_axes",
            perf_odd_shapes_and_axes,
            ids=list(map(idgen, perf_odd_shapes_and_axes)),
        )


def check_errors(queue, shape_and_axes, inverse=False):
    dtype = numpy.int32

    shape, axes = shape_and_axes

    data = numpy.arange(product(shape)).reshape(shape).astype(dtype)
    data_dev = Array.from_host(queue, data)

    shift = FFTShift(data_dev, axes=axes)
    shiftc = shift.compile(queue.device)

    ref_func = numpy.fft.ifftshift if inverse else numpy.fft.fftshift

    shiftc(queue, data_dev, data_dev, inverse)
    res_ref = ref_func(data, axes=axes)

    assert diff_is_negligible(data_dev.get(queue), res_ref)


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_errors(queue, errors_shape_and_axes, inverse):
    check_errors(queue, errors_shape_and_axes, inverse)


def test_trivial(some_queue):
    """
    Checks that even if the axes set is trivial (product of lengths == 1),
    the transformations are still attached and executed.
    """
    dtype = numpy.complex64
    shape = (128, 1, 1, 128)
    axes = (1, 2)
    param = 4

    data = get_test_array(shape, dtype)
    data_dev = Array.from_host(some_queue, data)
    res_dev = Array.empty_like(some_queue.device, data_dev)

    shift = FFTShift(data_dev, axes=axes)
    scale = mul_param(data_dev, numpy.int32)
    shift.parameter.input.connect(scale, scale.output, input_prime=scale.input, param=scale.param)

    shiftc = shift.compile(some_queue.device)
    shiftc(some_queue, res_dev, data_dev, param)
    assert diff_is_negligible(res_dev.get(some_queue), data * param)


def test_unordered_axes(some_queue):
    check_errors(some_queue, ((40, 50, 60), (2, 0)))


def test_no_axes(some_queue):
    check_errors(some_queue, ((40, 50, 60), None))


def check_performance(queue, shape_and_axes):
    # TODO: check double performance

    dtype = numpy.dtype("complex64")

    shape, axes = shape_and_axes

    data = numpy.arange(product(shape)).reshape(shape).astype(dtype)
    data_dev = Array.from_host(queue.device, data)

    shift = FFTShift(data_dev, axes=axes)
    shiftc = shift.compile(queue.device)

    res_dev = Array.empty_like(queue.device, data)

    attempts = 10
    times = []
    for i in range(attempts):
        t1 = time.time()
        shiftc(queue, res_dev, data_dev)
        queue.synchronize()
        times.append(time.time() - t1)

    res_ref = numpy.fft.fftshift(data, axes=axes)
    assert diff_is_negligible(res_dev.get(queue), res_ref)

    return min(times), product(shape) * dtype.itemsize


@pytest.mark.perf
@pytest.mark.returns("GB/s")
def test_even_performance(queue, perf_even_shape_and_axes):
    return check_performance(queue, perf_even_shape_and_axes)


@pytest.mark.perf
@pytest.mark.returns("GB/s")
def test_odd_performance(queue, perf_odd_shape_and_axes):
    return check_performance(queue, perf_odd_shape_and_axes)
