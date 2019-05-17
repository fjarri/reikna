import itertools

import pytest

import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.cluda import tempalloc
from reikna import concatenate, roll
from reikna.helpers import product

from helpers import *
from pytest_threadgen import parametrize_thread_tuple, create_thread_in_tuple


TEST_DTYPES = [
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
    numpy.float32, numpy.float64,
    numpy.complex64, numpy.complex128]


@pytest.fixture
def thr_and_global_size(request):
    return create_thread_in_tuple(request)


def pair_thread_with_gs(metafunc, tp):
    global_sizes = [
        (100,), (2000,), (1153,),
        (10, 10), (150, 250), (137, 547),
        (7, 11, 13), (50, 100, 100), (53, 101, 101)]

    rem_ids = []
    vals = []

    for gs in global_sizes:

        # If the thread will not support these limits, skip
        mgs = tp.device_params.max_num_groups
        if len(gs) > len(mgs) or (len(mgs) > 2 and len(gs) > 2 and mgs[2] < gs[2]):
            continue

        rem_ids.append(str(gs))
        vals.append((gs,))

    return [tp] * len(vals), vals, rem_ids


def pytest_generate_tests(metafunc):
    if 'thr_and_global_size' in metafunc.funcargnames:
        parametrize_thread_tuple(metafunc, 'thr_and_global_size', pair_thread_with_gs)


def simple_thread_test(thr):
    shape = (1000,)
    dtype = numpy.float32

    a = get_test_array(shape, dtype)
    a_dev = thr.to_device(a)
    a_back = thr.from_device(a_dev)

    assert diff_is_negligible(a, a_back)


def test_create_new_thread(cluda_api):
    thr = cluda_api.Thread.create()
    simple_thread_test(thr)


def test_array_functionality(thr):
    """
    Checks additional functionality introduced by the generic Array class.
    """
    arr = thr.array(1024, numpy.float32)
    assert arr.thread is thr


def test_array_type_after_binop(thr):
    """
    Checks that a binary operation keeps the Reikna Array type
    and does not reset it to PyOpenCL/PyCUDA array.
    """
    arr = thr.array(1024, numpy.float32)
    arr2 = thr.array(1024, numpy.float32)

    assert type(arr + arr2) == type(arr)


def test_array_copy(thr):
    """
    Checks that .copy() copies the array, keeps the Reikna Array type
    and does not reset it to PyOpenCL/PyCUDA array.
    """
    arr = thr.to_device(get_test_array(1024, numpy.float32))
    arr2 = arr.copy()

    assert type(arr2) == type(arr)
    assert diff_is_negligible(arr.get(), arr2.get())


def test_transfers(thr):
    a = get_test_array(1024, numpy.float32)

    def to_device1(x):
        return thr.to_device(x)
    def to_device2(x):
        y = thr.empty_like(x)
        thr.to_device(x, dest=y)
        return y
    def from_device1(x):
        return x.get()
    def from_device2(x):
        return thr.from_device(x)
    def from_device3(x):
        y = numpy.empty(x.shape, x.dtype)
        thr.from_device(x, dest=y)
        return y
    def from_device4(x):
        y = thr.from_device(x, async_=True)
        thr.synchronize()
        return y
    def from_device5(x):
        y = numpy.empty(x.shape, x.dtype)
        thr.from_device(x, dest=y, async_=True)
        thr.synchronize()
        return y

    to_device = (to_device1, to_device2)
    from_device = (from_device1, from_device2, from_device3, from_device4, from_device5)

    for to_d, from_d in itertools.product(to_device, from_device):
        a_device = to_d(a)
        a_copy = thr.copy_array(a_device)
        a_back = from_d(a_copy)
        assert diff_is_negligible(a, a_back)


@pytest.mark.parametrize(
    "dtype", TEST_DTYPES,
    ids=[dtypes.normalize_type(dtype).name for dtype in TEST_DTYPES])
def test_dtype_support(thr, dtype):
    # Test passes if either thread correctly reports that it does not support given dtype,
    # or it successfully compiles kernel that operates with this dtype.

    N = 256

    if not thr.device_params.supports_dtype(dtype):
        pytest.skip()

    mul = functions.mul(dtype, dtype)
    div = functions.div(dtype, dtype)
    program = thr.compile(
    """
    KERNEL void test(
        GLOBAL_MEM ${ctype} *dest, GLOBAL_MEM ${ctype} *a, GLOBAL_MEM ${ctype} *b)
    {
      const SIZE_T i = get_global_id(0);
      ${ctype} temp = ${mul}(a[i], b[i]);
      dest[i] = ${div}(temp, b[i]);
    }
    """, render_kwds=dict(ctype=dtypes.ctype(dtype), dtype=dtype, mul=mul, div=div))

    test = program.test

    # we need results to fit even in unsigned char
    a = get_test_array(N, dtype, high=8)
    b = get_test_array(N, dtype, no_zeros=True, high=8)

    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    dest_dev = thr.empty_like(a_dev)
    test(dest_dev, a_dev, b_dev, global_size=N)
    assert diff_is_negligible(thr.from_device(dest_dev), a)


def test_find_local_size(thr_and_global_size):
    """
    Check that if None is passed as local_size, kernel can find some local_size to run with
    (not necessarily optimal).
    """

    thr, global_size = thr_and_global_size

    program = thr.compile(
    """
    KERNEL void test(GLOBAL_MEM int *dest)
    {
        const SIZE_T i = get_global_id(0) +
            get_global_id(1) * get_global_size(0) +
            get_global_id(2) * get_global_size(1) * get_global_size(0);
        dest[i] = i;
    }
    """)
    test = program.test
    dest_dev = thr.array(global_size, numpy.int32)
    test(dest_dev, global_size=global_size)

    assert diff_is_negligible(dest_dev.get().ravel(),
        numpy.arange(product(global_size)).astype(numpy.int32))


@pytest.mark.parametrize(
    'tempalloc_cls',
    [tempalloc.TrivialManager, tempalloc.ZeroOffsetManager],
    ids=['trivial', 'zero_offset'])
@pytest.mark.parametrize('pack', [False, True], ids=['no_pack', 'pack'])
def test_tempalloc(cluda_api, tempalloc_cls, pack):

    shape = (10000,)
    dtype = numpy.int32
    thr = cluda_api.Thread.create(temp_alloc=dict(
        cls=tempalloc_cls, pack_on_alloc=False))

    # Dependency graph for the test
    dependencies = dict(
        _temp0=[],
        _temp1=['_temp9', '_temp8', '_temp3', '_temp5', '_temp4', '_temp7', '_temp6', 'input'],
        _temp10=['output', '_temp7'],
        _temp11=['_temp7'],
        _temp2=['input'],
        _temp3=['_temp1', 'input'],
        _temp4=['_temp9', '_temp8', '_temp1', '_temp7', '_temp6'],
        _temp5=['_temp1'],
        _temp6=['_temp1', '_temp4'],
        _temp7=['_temp9', '_temp1', '_temp4', 'output', '_temp11', '_temp10'],
        _temp8=['_temp1', '_temp4'],
        _temp9=['_temp1', '_temp4', '_temp7'],
        input=['_temp1', '_temp3', '_temp2'],
        output=['_temp10', '_temp7'])

    program = thr.compile(
    """
    KERNEL void fill(GLOBAL_MEM ${ctype} *dest, ${ctype} val)
    {
      const SIZE_T i = get_global_id(0);
      dest[i] = val;
    }

    KERNEL void transfer(GLOBAL_MEM ${ctype} *dest, GLOBAL_MEM ${ctype} *src)
    {
      const SIZE_T i = get_global_id(0);
      dest[i] = src[i];
    }
    """, render_kwds=dict(ctype=dtypes.ctype(dtype)))
    fill = program.fill
    transfer = program.transfer

    arrays = {}
    transfer_dest = thr.array(shape, dtype)

    # Allocate temporary arrays with dependencies
    for name in sorted(dependencies.keys()):
        deps = dependencies[name]
        arr_deps = [arrays[d] for d in deps if d in arrays]
        arrays[name] = thr.temp_array(shape, dtype, dependencies=arr_deps)
        fill(arrays[name], dtype(0), global_size=shape)

    if pack:
        thr.temp_alloc.pack()

    # Fill arrays with zeros
    for name in sorted(dependencies.keys()):
        deps = dependencies[name]
        arr_deps = [arrays[d] for d in deps if d in arrays]
        arrays[name] = thr.temp_array(shape, dtype, dependencies=arr_deps)
        fill(arrays[name], dtype(0), global_size=shape)

    for i, name in enumerate(sorted(dependencies.keys())):
        val = dtype(i + 1)
        fill(arrays[name], val, global_size=shape)
        for dep in dependencies[name]:
            # CUDA does not support get() for GPUArray with custom buffers,
            # So we need to transfer the data to a normal array first.
            transfer(transfer_dest, arrays[dep], global_size=shape)
            assert (transfer_dest.get() != val).all()


def test_constant_memory(thr):
    global_size = 100

    arr1 = get_test_array(global_size, numpy.int32)
    arr2 = get_test_array(global_size * 2, numpy.float32)
    ref = ((arr2[:global_size] + arr2[global_size:]) * arr1).astype(numpy.float32)

    src = """
        KERNEL void test(GLOBAL_MEM float *dest)
        {
            const SIZE_T i = get_global_id(0);
            dest[i] = (arr2[i] + arr2[i + get_global_size(0)]) * arr1[i];
        }
        """
    constant_arrays = dict(
        arr1=arr1, # use the parameters of an existing array; the array itself is not copied!
        arr2=(global_size * 2, numpy.float32)
        )

    # Compiled-in constant arrays only supported on CUDA
    if thr.api.get_id() != cluda.cuda_id():
        with pytest.raises(ValueError):
            program = thr.compile(src, constant_arrays=constant_arrays)
        return

    program = thr.compile(src, constant_arrays=constant_arrays)
    test = program.test

    # TODO: program.constant.arr1.set(arr1)
    program.set_constant('arr1', arr1)
    program.set_constant('arr2', arr2)

    dest_dev = thr.array(global_size, numpy.float32)
    test(dest_dev, global_size=global_size)

    assert diff_is_negligible(dest_dev.get(), ref)


    # Use a program/kernel with different constant arrays
    arr1_2 = get_test_array(global_size, numpy.int32)
    arr2_2 = get_test_array(global_size * 2, numpy.float32)
    ref_2 = ((arr2_2[:global_size] + arr2_2[global_size:]) * arr1_2).astype(numpy.float32)

    program2 = thr.compile(src, constant_arrays=constant_arrays)
    test2 = program2.test
    program2.set_constant('arr1', arr1_2)
    program2.set_constant('arr2', arr2_2)
    test2(dest_dev, global_size=global_size)
    assert diff_is_negligible(dest_dev.get(), ref_2)


    # Check that the first kernel preserved the data in its constant arrays
    test(dest_dev, global_size=global_size)
    assert diff_is_negligible(dest_dev.get(), ref)


def test_constant_memory_static_kernel(thr):
    global_size = 100

    arr1 = get_test_array(global_size, numpy.int32)
    arr2 = get_test_array(global_size * 2, numpy.float32)
    ref = ((arr2[:global_size] + arr2[global_size:]) * arr1).astype(numpy.float32)

    src = """
        KERNEL void test(GLOBAL_MEM float *dest)
        {
            const SIZE_T i = get_global_id(0);
            dest[i] = (arr2[i] + arr2[i + get_global_size(0)]) * arr1[i];
        }
        """
    constant_arrays = dict(
        arr1=arr1, # use the parameters of an existing array; the array itself is not copied!
        arr2=(global_size * 2, numpy.float32)
        )

    # Compiled-in constant arrays only supported on CUDA
    if thr.api.get_id() != cluda.cuda_id():
        with pytest.raises(ValueError):
            program = thr.compile_static(
                src, "test", global_size=global_size, constant_arrays=constant_arrays)
        return

    test = thr.compile_static(src, "test", global_size=global_size, constant_arrays=constant_arrays)

    # TODO: test.constant.arr1.set(arr1)
    test.set_constant('arr1', arr1)
    test.set_constant('arr2', thr.to_device(arr2)) # test that a device array is also accepted

    dest_dev = thr.array(global_size, numpy.float32)
    test(dest_dev)

    assert diff_is_negligible(dest_dev.get(), ref)


def test_offsets_in_kernel(thr):
    """
    Check that kernels receive the base data of arrays and have to add offsets manually.
    """

    global_size = 100
    dest_offset = 4
    src_offset = 2
    dtype = dtypes.normalize_type(numpy.int32)

    program = thr.compile("""
        KERNEL void test(GLOBAL_MEM int *dest, GLOBAL_MEM int *src)
        {
            const SIZE_T i = get_global_id(0);
            dest[i + ${dest_offset}] = src[i + ${src_offset}];
        }
        """,
        render_kwds=dict(dest_offset=dest_offset, src_offset=src_offset))
    test = program.test

    dest_dev_base = thr.array(global_size + dest_offset, dtype)
    dest_dev = thr.array(
        global_size, dtype, offset=dest_offset * dtype.itemsize, base=dest_dev_base)

    src_base = numpy.arange(global_size + src_offset).astype(dtype)
    src_dev_base = thr.to_device(src_base)
    src_dev = thr.array(global_size, dtype, offset=src_offset * dtype.itemsize, base=src_dev_base)

    test(dest_dev, src_dev, global_size=global_size)
    dest_ref = src_base[src_offset:]

    assert diff_is_negligible(dest_dev.get(), dest_ref)


class _GetSlices:
    def __getitem__(self, index):
        return index

_get_slices = _GetSlices()

_setitem_view_tests = [
    (_get_slices[:, :], 1, "[:,:]=scalar"),
    (_get_slices[:, 1], 1, "[:,i]=scalar"),
    (_get_slices[1, :], 1, "[i,:]=scalar"),
    (_get_slices[1, 2], 1, "[i,i]=scalar"),
    (_get_slices[:, :], numpy.arange(10 * 20).reshape(10, 20), "[:,:]=array"),
    (_get_slices[:, 2], numpy.arange(10), "[:,i]=array"),
    (_get_slices[2, :], numpy.arange(20), "[i,:]=array"),
]

@pytest.mark.parametrize(
    'setitem_test',
    [test[:2] for test in _setitem_view_tests],
    ids=[test[2] for test in _setitem_view_tests])
def test_setitem_view(thr, setitem_test):
    data = numpy.arange(10 * 20).reshape(10, 20).astype(numpy.int32)
    data_dev = thr.to_device(data)

    slices, value = setitem_test

    data[slices] = value
    data_dev[slices] = value

    assert diff_is_negligible(data_dev.get(), data)


# A regression test for a 0-dim array as a source for __setitem__
def test_setitem_0dim(thr):
    data = numpy.arange(10 * 20).reshape(10, 20).astype(numpy.int32)
    data_dev = thr.to_device(data)

    data[1, 2] = data[2, 3]
    data_dev[1, 2] = data_dev[2, 3]

    assert diff_is_negligible(data_dev.get(), data)


_get_view_tests = [
    (_get_slices[:, :], "[:,:]"),
    (_get_slices[:, 1], "[:,i]"),
    (_get_slices[1, :], "[i,:]"),
    (_get_slices[1, 2], "[i,i]"),
]

@pytest.mark.parametrize(
    'get_test_slices',
    [test[0] for test in _get_view_tests],
    ids=[test[1] for test in _get_view_tests])
def test_get_view(thr, get_test_slices):
    data = numpy.arange(10 * 20).reshape(10, 20).astype(numpy.int32)
    data_dev = thr.to_device(data)

    view_ref = data[get_test_slices]
    view_dev = data_dev[get_test_slices]

    assert diff_is_negligible(view_dev.get(), view_ref)


def test_concatenate(some_thr):

    template_shape = [10, None, 30]
    axis = 1
    dims = [3, 5, 10]

    arrays = []
    for i in range(len(dims)):
        template_shape[axis] = dims[i]
        arrays.append(get_test_array(tuple(template_shape), numpy.int32))

    ref = numpy.concatenate(arrays, axis=axis)

    arrays_dev = [some_thr.to_device(array) for array in arrays]
    test = concatenate(arrays_dev, axis=axis)

    assert diff_is_negligible(test.get(), ref)


@pytest.mark.parametrize('shift', [-10, 0, 11])
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_roll(some_thr, shift, axis):

    array = get_test_array((5, 6, 7), numpy.int32)
    array_dev = some_thr.to_device(array)
    ref = numpy.roll(array, shift, axis=axis)
    test = roll(array_dev, shift, axis=axis)

    assert diff_is_negligible(test.get(), ref)


@pytest.mark.parametrize('shift', [-1, 0, 1])
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_roll_method(some_thr, shift, axis):

    array = get_test_array((5, 6, 7), numpy.int32)
    array = numpy.arange(12).reshape(3, 4).astype(numpy.int32)
    array_dev = some_thr.to_device(array)
    ref = numpy.roll(array, shift, axis=axis)
    array_dev.roll(shift, axis=axis)

    assert diff_is_negligible(array_dev.get(), ref)
