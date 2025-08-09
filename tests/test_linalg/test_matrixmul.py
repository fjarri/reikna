import time

import numpy
import pytest
from grunnur import Array, ArrayMetadata, dtypes

from helpers import *
from reikna.helpers import product
from reikna.linalg import MatrixMul


def pytest_generate_tests(metafunc):
    if "perf_bwo" in metafunc.fixturenames:
        bwos = [8, 16, 32]
        ids = ["8x8", "16x16", "32x32"]
        metafunc.parametrize("perf_bwo", bwos, ids=ids)

    if "perf_shape" in metafunc.fixturenames:
        mem_limit = 2**20
        sizes = [16, 32, 64, 256, 512, 25]

        perf_shapes = []

        for size in sizes:
            perf_shapes.append((mem_limit // size**2, size))

        ids = []
        for batch, size in perf_shapes:
            ids.append(str(batch) + "x" + str(size) + "," + str(size))

        metafunc.parametrize("perf_shape", perf_shapes, ids=ids)

    if "sizes" in metafunc.fixturenames:
        sizes = [(15, 17, 3), (122, 5, 1000), (45, 99, 40), (56, 78, 44)]
        ids = [str(size) for size in sizes]
        metafunc.parametrize("sizes", sizes, ids=ids)

    if "batches" in metafunc.fixturenames:
        batches = [(tuple(), tuple()), (tuple(), (14,)), ((35,), tuple()), ((12,), (12,))]
        ids = [str(batch) for batch in batches]
        metafunc.parametrize("batches", batches, ids=ids)

    if "transposed_a" in metafunc.fixturenames:
        metafunc.parametrize("transposed_a", [False, True], ids=["A", "A.T"])

    if "transposed_b" in metafunc.fixturenames:
        metafunc.parametrize("transposed_b", [False, True], ids=["B", "B.T"])

    if "arg_dtypes" in metafunc.fixturenames:
        arg_dtypes = [(False, False), (False, True), (True, False), (True, True)]
        mark = lambda x: "c" if x else "r"
        ids = [mark(t1) + mark(t2) for t1, t2 in arg_dtypes]
        metafunc.parametrize("arg_dtypes", arg_dtypes, ids=ids)


def ref_dot(a, b):
    a_batch = product(a.shape[:-2])
    b_batch = product(b.shape[:-2])

    assert a_batch == b_batch or a_batch == 1 or b_batch == 1

    a = a.reshape(a_batch, a.shape[-2], a.shape[-1])
    b = b.reshape(b_batch, b.shape[-2], b.shape[-1])

    out_batch = max(a_batch, b_batch)
    out_shape = (out_batch, a.shape[-2], b.shape[-1])
    out_dtype = numpy.result_type(a.dtype, b.dtype)
    out = numpy.empty(out_shape, out_dtype)

    for i in range(out_batch):
        ai = 0 if a_batch == 1 else i
        bi = 0 if b_batch == 1 else i
        out[i] = numpy.dot(a[ai], b[bi])

    if a_batch == b_batch == 1:
        out = out.reshape(out.shape[-2], out_shape[-1])

    return out


def transpose(m):
    axes = list(range(len(m.shape)))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return m.transpose(*axes)


def check_errors(queue, a_shape, a_dtype, b_shape, b_dtype, transposed_a=False, transposed_b=False):
    a = get_test_array(a_shape, a_dtype)
    b = get_test_array(b_shape, b_dtype)

    a_ref = transpose(a) if transposed_a else a
    b_ref = transpose(b) if transposed_b else b

    res_ref = ref_dot(a_ref, b_ref)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.from_host(queue, b)
    res_dev = Array.empty_like(queue.device, res_ref)

    dot = MatrixMul(
        a_dev, b_dev, out_arr_t=res_dev, transposed_a=transposed_a, transposed_b=transposed_b
    )
    dotc = dot.compile(queue.device)
    dotc(queue, res_dev, a_dev, b_dev)

    assert diff_is_negligible(res_dev.get(queue), res_ref)


def test_shapes(queue, batches, sizes):
    a_size, convolution_size, b_size = sizes
    a_batch, b_batch = batches

    a_shape = (a_size, convolution_size)
    b_shape = (convolution_size, b_size)

    check_errors(queue, a_batch + a_shape, numpy.float32, b_batch + b_shape, numpy.float32)


def test_transposed(queue, sizes, transposed_a, transposed_b):
    a_size, convolution_size, b_size = sizes
    a_batch = (10,)
    b_batch = (10,)

    if transposed_a:
        a_shape = (convolution_size, a_size)
    else:
        a_shape = (a_size, convolution_size)

    if transposed_b:
        b_shape = (b_size, convolution_size)
    else:
        b_shape = (convolution_size, b_size)

    check_errors(
        queue,
        a_batch + a_shape,
        numpy.float32,
        b_batch + b_shape,
        numpy.float32,
        transposed_a=transposed_a,
        transposed_b=transposed_b,
    )


def test_dtypes(queue, arg_dtypes):
    c1, c2 = arg_dtypes

    dtype = numpy.float32
    dtype1 = dtypes.complex_for(dtype) if c1 else dtype
    dtype2 = dtypes.complex_for(dtype) if c2 else dtype

    check_errors(queue, (30, 40, 50), dtype1, (30, 50, 60), dtype2)


def test_out_arr_shape():
    a = ArrayMetadata((1, 22, 33), numpy.float32)
    b = ArrayMetadata((2, 3, 33, 44), numpy.float32)
    dot = MatrixMul(a, b)
    assert dot.parameter.output.shape == (2, 3, 22, 44)


def check_performance(queue, perf_shape, bwo=None, transposed_a=False, transposed_b=False):
    # TODO: check double performance

    dtype = numpy.float32
    batch, size = perf_shape

    shape = (batch, size, size)

    a = get_test_array(shape, dtype)
    b = get_test_array(shape, dtype)

    a_ref = transpose(a) if transposed_a else a
    b_ref = transpose(b) if transposed_b else b

    a_dev = Array.from_host(queue.device, a)
    b_dev = Array.from_host(queue.device, b)
    res_ref = ref_dot(a_ref, b_ref)
    res_dev = Array.empty(queue.device, res_ref.shape, dtype=dtype)

    dot = MatrixMul(
        a_dev,
        b_dev,
        out_arr_t=res_dev,
        block_width_override=bwo,
        transposed_a=transposed_a,
        transposed_b=transposed_b,
    )

    try:
        dotc = dot.compile(queue.device)
    except ValueError:
        pytest.skip()

    attempts = 10
    times = []
    for i in range(attempts):
        t1 = time.time()
        dotc(queue, res_dev, a_dev, b_dev)
        queue.synchronize()
        times.append(time.time() - t1)

    assert diff_is_negligible(res_dev.get(queue), res_ref)

    return min(times), batch * size**3 * 2


@pytest.mark.perf
@pytest.mark.returns("GFLOPS")
def test_performance_shape(queue, perf_shape, transposed_a, transposed_b):
    return check_performance(
        queue, perf_shape, transposed_a=transposed_a, transposed_b=transposed_b
    )


@pytest.mark.perf
@pytest.mark.returns("GFLOPS")
def test_performance_block_width(queue, perf_bwo):
    return check_performance(queue, (4, 512), bwo=perf_bwo)
