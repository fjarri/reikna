import itertools
import time

import numpy
import pytest
from grunnur import Array, ArrayMetadata, dtypes

from helpers import *
from reikna.fft import FFT
from reikna.helpers import product
from reikna.transformations import mul_param


def pytest_generate_tests(metafunc):
    perf_log_shapes = [
        (4,),
        (10,),
        (13,),  # 1D
        (4, 4),
        (7, 7),
        (10, 10),  # 2D
        (4, 4, 4),
        (5, 5, 7),
        (7, 7, 7),
    ]  # 3D
    perf_mem_limit = 4 * 2**20

    if "local_shape_and_axes" in metafunc.fixturenames:

        def idgen(val):
            batch, size = val[0]
            return str(batch) + "x" + str(size)

        # These values are supposed to check all code paths in
        # fft.mako::insertGlobalLoadsAndTranspose.
        # Some of them will probably become global FFTs on lower-end GPUs, but that's fine.
        #
        # We need to try different FFT sizes in order to catch the path where
        # 1. ``threads_per_xform >= mem_coalesce_width``
        #    (first code path, corresponds to relatively large FFT sizes)
        # 2. ``xforms_per_workgroup`` still > 1
        #    (which will allow us to have ``xforms_remainder != 0``)

        mem_limit = 2**20
        size_powers = (3, 7, 8, 9, 10, 12)

        vals = []
        for p in size_powers:
            size = 2**p
            batch = mem_limit // size

            # "size / 2 - 1" will lead to full FFT of size ``size``
            # in the current version of Bluestein's algorithm
            pad_test_size = size // 2 - 1

            for s in (size, pad_test_size):
                # testing batch = 1, non-multiple batch and power of 2 batch
                for b in (1, batch - 1, batch):
                    vals.append(((b, s), (1,)))

        metafunc.parametrize("local_shape_and_axes", vals, ids=list(map(idgen, vals)))

    elif "global_shape_and_axes" in metafunc.fixturenames:

        def idgen(val):
            outer_batch, size, inner_batch = val[0]
            return str(outer_batch) + "x" + str(size) + "x" + str(inner_batch)

        # These values are supposed to check all code paths in
        # fft.mako::fft_global.
        mem_limit = 2**22
        size_powers = (3, 7, 9, 10)

        vals = []
        for p in size_powers:
            size = 2**p
            batch = mem_limit // size // 64

            # "size / 2 - 1" will lead to full FFT of size ``size``
            # in the current version of Bluestein's algorithm
            pad_test_size = size // 2 - 1

            for s in (size, pad_test_size):
                # possible inner batches: small power of 2, small non-power-of-2,
                # big power of 2 (more than coalesce_width), big non-power-of-2
                for ib in (2, 3, 63, 64):
                    for ob in (1, batch - 1, batch):
                        vals.append(((ob, s, ib), (1,)))

        # big size (supposed to be > local_kernel_limit * MAX_RADIX)
        big_size = 2**15
        batch = mem_limit // big_size // 8
        vals.append(((1, big_size, 1), (1,)))
        vals.append(((1, big_size, 2), (1,)))
        vals.append(((1, big_size, 3), (1,)))
        vals.append(((batch, big_size, 3), (1,)))
        vals.append(((batch - 1, big_size, 3), (1,)))

        metafunc.parametrize("global_shape_and_axes", vals, ids=list(map(idgen, vals)))

    elif "sequence_shape_and_axes" in metafunc.fixturenames:

        def idgen(non2problem_shape_and_axes):
            shape, axes = non2problem_shape_and_axes
            return str(shape) + "over" + str(axes)

        vals = [
            ((3, 255, 7), (2, 0)),
            ((17, 200, 131), (1, 0)),
            ((7, 1000, 11), (1, 2)),
            ((15, 900, 57), (2, 0, 1)),
        ]

        metafunc.parametrize("sequence_shape_and_axes", vals, ids=list(map(idgen, vals)))

    elif "perf_shape_and_axes" in metafunc.fixturenames:
        vals = []
        ids = []
        for log_shape in perf_log_shapes:
            shape = tuple(2**x for x in log_shape)
            batch = perf_mem_limit // (2 ** sum(log_shape))
            vals.append(((batch,) + shape, tuple(range(1, len(shape) + 1))))
            ids.append(str(batch) + "x" + str(shape))

        metafunc.parametrize("perf_shape_and_axes", vals, ids=ids)

    elif "non2problem_perf_shape_and_axes" in metafunc.fixturenames:
        vals = []
        ids = []
        for log_shape in perf_log_shapes:
            for modifier in (1, -1):
                shape = tuple(2 ** (x - 1) + modifier for x in log_shape)
                batch = perf_mem_limit // (2 ** sum(log_shape))
                vals.append(((batch,) + shape, tuple(range(1, len(shape) + 1))))
                ids.append(str(batch) + "x" + str(shape))

        metafunc.parametrize("non2problem_perf_shape_and_axes", vals, ids=ids)


def test_typecheck():
    with pytest.raises(ValueError, match="FFT computation requires array of a complex dtype"):
        fft = FFT(ArrayMetadata(shape=100, dtype=numpy.float32))


# Since we're using single precision for tests (for compatibility reasons),
# the FFTs (especially non-power-of-2 ones) are not very accurate
# (GPUs historically tend to cut corners in single precision).
# So we're lowering tolerances when comparing to the reference in these tests.
def check_errors(queue, shape_and_axes, atol=2e-5, rtol=1e-3):
    dtype = numpy.complex64

    shape, axes = shape_and_axes

    data = get_test_array(shape, dtype)
    data_dev = Array.from_host(queue, data)

    fft = FFT(data_dev, axes=axes)
    fftc = fft.compile(queue.device)

    # forward transform
    # Testing inplace transformation, because if this works,
    # then the out of place one will surely work too.
    fftc(queue, data_dev, data_dev)
    fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(data_dev.get(queue), fwd_ref, atol=atol, rtol=rtol)

    # inverse transform
    data_dev = Array.from_host(queue, data)
    fftc(queue, data_dev, data_dev, inverse=True)
    inv_ref = numpy.fft.ifftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(data_dev.get(queue), inv_ref, atol=atol, rtol=rtol)


def test_trivial(some_queue):
    """
    Checks that even if the FFT is trivial (problem size == 1),
    the transformations are still attached and executed.
    """
    dtype = numpy.complex64
    shape = (128, 1, 1, 128)
    axes = (1, 2)
    param = 4

    data = get_test_array(shape, dtype)
    data_dev = Array.from_host(some_queue, data)
    res_dev = Array.empty_like(some_queue.device, data_dev)

    fft = FFT(data_dev, axes=axes)
    scale = mul_param(data_dev, numpy.int32)
    fft.parameter.input.connect(scale, scale.output, input_prime=scale.input, param=scale.param)

    fftc = fft.compile(some_queue.device)
    fftc(some_queue, res_dev, data_dev, param)
    assert diff_is_negligible(res_dev.get(some_queue), data * param)


def test_local(queue, local_shape_and_axes):
    check_errors(queue, local_shape_and_axes)


def test_global(queue, global_shape_and_axes):
    check_errors(queue, global_shape_and_axes)


def test_sequence(queue, sequence_shape_and_axes):
    # This test is particularly sensitive to inaccuracies in single precision,
    # hence the particularly high tolerance.
    check_errors(queue, sequence_shape_and_axes, rtol=1e-2)


def check_performance(queue, shape_and_axes, fast_math):
    # TODO: check double performance

    shape, axes = shape_and_axes
    dtype = numpy.dtype("complex64")

    data = get_test_array(shape, dtype)
    data_dev = Array.from_host(queue.device, data)
    res_dev = Array.empty_like(queue.device, data_dev)

    fft = FFT(data_dev, axes=axes)
    fftc = fft.compile(queue.device, fast_math=fast_math)

    attempts = 10
    times = []
    for i in range(attempts):
        t1 = time.time()
        fftc(queue, res_dev, data_dev)
        queue.synchronize()
        times.append(time.time() - t1)

    fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
    assert diff_is_negligible(res_dev.get(queue), fwd_ref, atol=1e-2, rtol=1e-4)

    return min(times), product(shape) * sum([numpy.log2(shape[a]) for a in axes]) * 5


@pytest.mark.perf
@pytest.mark.returns("GFLOPS")
def test_power_of_2_performance(queue, perf_shape_and_axes, fast_math):
    return check_performance(queue, perf_shape_and_axes, fast_math)


@pytest.mark.perf
@pytest.mark.returns("GFLOPS")
def test_non_power_of_2_performance(queue, non2problem_perf_shape_and_axes, fast_math):
    return check_performance(queue, non2problem_perf_shape_and_axes, fast_math)
