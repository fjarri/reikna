import itertools

import pytest

from reikna.cluda import OutOfResourcesError
import reikna.cluda.vsize as vsize
from reikna.helpers import min_blocks, product
import reikna.cluda.dtypes as dtypes
from helpers import *

from pytest_threadgen import parametrize_thread_tuple, create_thread_in_tuple


vals_find_local_size = [
    ((40, 50), 1, (1, 1)),
    ((7, 11, 13), 1001, (7, 11, 13)),
    ((100,), 6, (6,)),
    ((3, 15), 12, (3, 4)),
    ((10, 10, 10, 10, 10), 32, (2, 2, 2, 2, 2)),
    ((16, 16, 128, 1024, 2), 30 * 32, (16, 4, 3, 5, 1))
    ]
@pytest.mark.parametrize(
    ('global_size', 'flat_local_size', 'expected_local_size'),
    vals_find_local_size,
    ids=[str(x[:2]) for x in vals_find_local_size])
def test_find_local_size(global_size, flat_local_size, expected_local_size):
    """
    Checking that ``find_local_size`` finds the sizes we expect from it.
    """
    local_size = vsize.find_local_size(global_size, flat_local_size)
    assert product(local_size) == flat_local_size
    assert local_size == expected_local_size


vals_group_dimensions = [
    ((1, 1, 1), (2, 2)),
    ((1, 2, 3), (2, 3)),
    ((16, 16), (7, 45)),
    ((16, 16), (7, 45, 50)),
    ((3, 4, 5), (72,)),
    ((3, 4, 5), (4, 5, 6)),
    ((134, 1, 1), (9, 5, 3)),
    ((10, 1, 1, 10, 1, 1, 10, 1, 1), (100, 100))
    ]
@pytest.mark.parametrize(
    ('virtual_shape', 'available_shape'),
    vals_group_dimensions,
    ids=[str(x) for x in vals_group_dimensions])
def test_group_dimensions(virtual_shape, available_shape):
    """
    Tests that ``group_dimensions()`` obeys its contracts.
    """
    v_groups, a_groups = vsize.group_dimensions(virtual_shape, available_shape)
    v_dims = []
    a_dims = []
    for v_group, a_group in zip(v_groups, a_groups):

        # Check that axis indices in groups are actually in range
        assert any(vdim < len(virtual_shape) for vdim in v_group)
        assert any(adim < len(available_shape) for adim in a_group)

        # Check that the total number of elements (threads) in the virtual group
        # is not greater than the number of elements in the real group
        v_shape = virtual_shape[v_group[0]:v_group[-1]+1]
        a_shape = available_shape[a_group[0]:a_group[-1]+1]
        assert(product(v_shape) <= product(a_shape))

        v_dims += v_group
        a_dims += a_group

    # Check that both virtual and real groups axes add up to a successive list
    # without intersections.
    assert v_dims == list(range(len(virtual_shape)))
    assert a_dims == list(range(len(available_shape[:len(a_dims)])))


vals_find_bounding_shape = [
    (256, (15, 18)),
    (256, (10, 3, 10)),
    (299, (10, 3, 10))
    ]
@pytest.mark.parametrize(
    ('virtual_size', 'available_shape'),
    vals_find_bounding_shape,
    ids=[str(x) for x in vals_find_bounding_shape])
def test_find_bounding_shape(virtual_size, available_shape):
    """
    Tests that ``find_bounding_shape()`` obeys its contracts.
    """
    shape = vsize.find_bounding_shape(virtual_size, available_shape)
    assert all(isinstance(d, int) for d in shape)
    assert product(shape) >= virtual_size
    assert all(d <= ad for d, ad in zip(shape, available_shape))


def pytest_generate_tests(metafunc):
    if 'testvs' in metafunc.funcargnames:
        global_sizes = [
            (35,), (31*31*4,),
            (15, 13), (13, 35),
            (17, 15, 13), (5, 33, 75), (10, 10, 10, 5)]
        local_sizes = [None, (4,), (2, 4), (6, 4, 2)]
        mngs = [(26, 31), (34, 56, 25)]
        mwiss = [(4, 4), (5, 3), (9, 5, 3)]

        vals = []

        for gs, ls, mng, mwis in itertools.product(global_sizes, local_sizes, mngs, mwiss):
            testvs = VirtualSizesHelper.try_create(gs, ls, mng, mwis)
            if testvs is None:
                continue
            vals.append(testvs)

        metafunc.parametrize('testvs', vals, ids=[str(x) for x in vals])

    if 'incorrect_testvs' in metafunc.funcargnames:
        vals = [
            # Bounding global size (32, 32) is too big for the grid limit and given block size
            VirtualSizesHelper((32, 32), (4, 4), (7, 8), (4, 4)),
            # Local size is too big
            VirtualSizesHelper((32, 32), (5, 4), (16, 16), (4, 4)),
            ]
        metafunc.parametrize('incorrect_testvs', vals, ids=[str(x) for x in vals])


class ReferenceIds:

    def __init__(self, global_size, local_size):
        self.global_size = global_size
        if local_size is not None:
            self.local_size = local_size
            self.grid_size = tuple(min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))

    def _tile_pattern(self, pattern, axis, full_shape):

        pattern_shape = [x if i == axis else 1 for i, x in enumerate(full_shape)]
        pattern = pattern.reshape(*pattern_shape)

        tile_shape = [x if i != axis else 1 for i, x in enumerate(full_shape)]
        pattern = numpy.tile(pattern, tile_shape)

        return pattern.astype(numpy.int32)

    def predict_local_ids(self, dim):
        global_len = self.global_size[dim]
        local_len = self.local_size[dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.tile(numpy.arange(local_len), repetitions)[:global_len]
        return self._tile_pattern(pattern, dim, self.global_size)

    def predict_group_ids(self, dim):
        global_len = self.global_size[dim]
        local_len = self.local_size[dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.repeat(numpy.arange(repetitions), local_len)[:global_len]
        return self._tile_pattern(pattern, dim, self.global_size)

    def predict_global_ids(self, dim):
        global_len = self.global_size[dim]

        pattern = numpy.arange(global_len)
        return self._tile_pattern(pattern, dim, self.global_size)


class VirtualSizesHelper:

    @classmethod
    def try_create(cls, global_size, local_size, max_num_groups, max_work_item_sizes):
        """
        This method is used to filter working combinations of parameters
        from the cartesian product of all possible ones.
        Returns ``None`` if the parameters are not compatible.
        """
        if len(max_num_groups) != len(max_work_item_sizes):
            return None

        if local_size is not None:
            if len(local_size) > len(global_size):
                return None
            else:
                # we need local size and global size of the same length
                local_size = local_size + (1,) * (len(global_size) - len(local_size))

            if product(local_size) > product(max_work_item_sizes):
                return None

            bounding_global_size = [
                ls * min_blocks(gs, ls) for gs, ls
                in zip(global_size, local_size)]

            if product(bounding_global_size) > product(max_num_groups):
                return None

        else:
            if product(global_size) > product(max_num_groups):
                return None

        return cls(global_size, local_size, max_num_groups, max_work_item_sizes)

    def __init__(self, global_size, local_size, max_num_groups, max_work_item_sizes):
        self.global_size = global_size
        self.local_size = local_size
        if local_size is not None:
            self.grid_size = tuple(min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))

        self.max_num_groups = max_num_groups
        self.max_work_item_sizes = max_work_item_sizes

    def is_supported_by(self, thr):
        return (
            len(self.max_num_groups) <= len(thr.device_params.max_num_groups) and
            all(ng <= mng for ng, mng
                in zip(self.max_num_groups, thr.device_params.max_num_groups)) and
            len(self.max_work_item_sizes) <= len(thr.device_params.max_work_item_sizes) and
            all(wi <= mwi for wi, mwi
                in zip(self.max_work_item_sizes, thr.device_params.max_work_item_sizes)))

    def __str__(self):
        return "{gs}-{ls}-limited-by-{mng}-{mwis}".format(
            gs=self.global_size, ls=self.local_size,
            mng=self.max_num_groups, mwis=self.max_work_item_sizes)


class override_device_params:
    """
    Some of the tests here need to make thread/workgroup number limits
    in DeviceParameters of the thread lower, so that they are easier to test.

    This context manager hacks into the Thread and replaces the ``device_params`` attribute.
    Since threads are reused, the old device_params must be restored on exit.
    """

    def __init__(self, thr, **kwds):
        self._thr = thr
        self._kwds = kwds

    def __enter__(self):
        self._old_device_params = self._thr.device_params
        device_params = self._thr.api.DeviceParameters(self._thr._device)
        for kwd, val in self._kwds.items():
            setattr(device_params, kwd, val)
        self._thr.device_params = device_params
        return self._thr

    def __exit__(self, *args):
        self._thr.device_params = self._old_device_params


def test_ids(thr, testvs):
    """
    Test that virtual IDs are correct for each thread.
    """
    if not testvs.is_supported_by(thr):
        pytest.skip()

    ref = ReferenceIds(testvs.global_size, testvs.local_size)

    with override_device_params(
            thr, max_num_groups=testvs.max_num_groups,
            max_work_item_sizes=testvs.max_work_item_sizes, warp_size=2) as limited_thr:

        get_ids = limited_thr.compile_static("""
        KERNEL void get_ids(
            GLOBAL_MEM int *local_ids,
            GLOBAL_MEM int *group_ids,
            GLOBAL_MEM int *global_ids,
            int vdim)
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_flat_id();
            local_ids[i] = virtual_local_id(vdim);
            group_ids[i] = virtual_group_id(vdim);
            global_ids[i] = virtual_global_id(vdim);
        }
        """, 'get_ids', testvs.global_size, local_size=testvs.local_size)

    local_ids = thr.array(ref.global_size, numpy.int32)
    group_ids = thr.array(ref.global_size, numpy.int32)
    global_ids = thr.array(ref.global_size, numpy.int32)

    for vdim in range(len(testvs.global_size)):

        get_ids(local_ids, group_ids, global_ids, numpy.int32(vdim))

        assert diff_is_negligible(global_ids.get(), ref.predict_global_ids(vdim))
        if testvs.local_size is not None:
            assert diff_is_negligible(local_ids.get(), ref.predict_local_ids(vdim))
            assert diff_is_negligible(group_ids.get(), ref.predict_group_ids(vdim))


def test_sizes(thr, testvs):
    """
    Test that virtual sizes are correct.
    """
    if not testvs.is_supported_by(thr):
        pytest.skip()

    ref = ReferenceIds(testvs.global_size, testvs.local_size)
    vdims = len(testvs.global_size)

    with override_device_params(
        thr, max_num_groups=testvs.max_num_groups,
        max_work_item_sizes=testvs.max_work_item_sizes, warp_size=2) as limited_thr:

        get_sizes = limited_thr.compile_static("""
        KERNEL void get_sizes(GLOBAL_MEM int *sizes)
        {
            if (virtual_global_id(0) > 0) return;

            for (int i = 0; i < ${vdims}; i++)
            {
                sizes[i] = virtual_local_size(i);
                sizes[i + ${vdims}] = virtual_num_groups(i);
                sizes[i + ${vdims * 2}] = virtual_global_size(i);
            }
            sizes[${vdims * 3}] = virtual_global_flat_size();
        }
        """,
            'get_sizes',
            testvs.global_size, local_size=testvs.local_size,
            render_kwds=dict(vdims=vdims))

    sizes = thr.array(vdims * 3 + 1, numpy.int32)
    get_sizes(sizes)

    sizes = sizes.get()
    local_sizes = sizes[0:vdims]
    grid_sizes = sizes[vdims:vdims*2]
    global_sizes = sizes[vdims*2:vdims*3]
    flat_size = sizes[vdims*3]

    global_sizes_ref = numpy.array(testvs.global_size).astype(numpy.int32)
    assert diff_is_negligible(global_sizes, global_sizes_ref)
    assert flat_size == product(testvs.global_size)

    if testvs.local_size is not None:
        grid_sizes_ref = numpy.array(testvs.grid_size).astype(numpy.int32)
        assert diff_is_negligible(grid_sizes, grid_sizes_ref)
        local_sizes_ref = numpy.array(testvs.local_size).astype(numpy.int32)
        assert diff_is_negligible(local_sizes, local_sizes_ref)


def test_incorrect_sizes(thr, incorrect_testvs):
    """
    Test that for sizes which exceed the thread capability, the exception is raised.
    """
    with override_device_params(
        thr, max_num_groups=incorrect_testvs.max_num_groups,
        max_work_item_sizes=incorrect_testvs.max_work_item_sizes, warp_size=2) as limited_thr:

        with pytest.raises(OutOfResourcesError):
            kernel = thr.compile_static("""
            KERNEL void test(GLOBAL_MEM int *temp)
            {
                temp[0] = 1;
            }
            """, 'test', incorrect_testvs.global_size, local_size=incorrect_testvs.local_size)
