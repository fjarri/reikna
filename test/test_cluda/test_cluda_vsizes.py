import itertools

import pytest

import reikna.cluda as cluda
from reikna.cluda.vsize import find_local_size
from reikna.helpers import min_blocks, product
import reikna.cluda.dtypes as dtypes
from helpers import *

from pytest_threadgen import parametrize_thread_tuple, create_thread_in_tuple


pytest_funcarg__thr_with_gs_limits = create_thread_in_tuple


vals_find_local_size = [
    ((7, 11, 13), 1001, (7, 11, 13)),
    ((100,), 6, (6,)),
    ((3, 15), 12, (3, 4)),
    ((10, 10, 10, 10, 10), 32, (2, 2, 2, 2, 2)),
    ((2, 1024, 128, 16, 16), 30 * 32, (2, 96, 5, 1, 1))
    ]
@pytest.mark.parametrize(
    ('global_size', 'flat_local_size', 'expected_local_size'),
    vals_find_local_size,
    ids=[str(x[:2]) for x in vals_find_local_size])
def test_find_local_size(global_size, flat_local_size, expected_local_size):
    """
    Checking that ``find_local_size`` finds the sizes we expect from it.
    """
    local_size = find_local_size(global_size, flat_local_size)
    assert product(local_size) == flat_local_size
    assert local_size == expected_local_size


def set_thread_gs_limits(metafunc, tc):
    """
    Parametrize threads with small grid limits for testing purposes
    """
    new_tcs = []
    rem_ids = []
    for gl in [[31, 31], [31, 63], [31, 31, 31]]:

        # If the thread will not support these limits, skip
        thr = tc()
        mgs = thr.device_params.max_num_groups
        del thr
        if len(gl) > len(mgs) or any(g > mg for g, mg in zip(gl, mgs)):
            continue

        # New thread creator function
        gl_local = gl # otherwise the closure will use the last 'gl' value the cycle reaches
        def new_tc():
            thr = tc()
            thr.override_device_params(max_num_groups=gl_local)
            return thr

        rem_ids.append(str(gl))
        new_tcs.append(new_tc)

    return new_tcs, [tuple()] * len(new_tcs), rem_ids


def pytest_generate_tests(metafunc):
    if 'testvs' in metafunc.funcargnames:
        grid_sizes = [
            (13,), (35,), (31*31*4,),
            (13, 15), (35, 13),
            (13, 15, 17), (75, 33, 5)]
        local_sizes = [None, (4,), (4, 4), (4, 4, 4)]
        mngs = [(31, 31), (31, 63), (31, 31, 31)]
        mwiss = [(2, 2), (4, 4), (3, 5), (3, 5, 9)]

        vals = []

        for gs, ls, mng, mwis in itertools.product(grid_sizes, local_sizes, mngs, mwiss):
            testvs = TestVirtualSizes.try_create(gs, ls, mng, mwis)
            if testvs is None:
                continue
            vals.append(testvs)

        metafunc.parametrize('testvs', vals, ids=[str(x) for x in vals])

    if 'thr_with_gs_limits' in metafunc.funcargnames:
        parametrize_thread_tuple(metafunc, 'thr_with_gs_limits', set_thread_gs_limits)
    if 'gs_is_multiple' in metafunc.funcargnames:
        metafunc.parametrize('gs_is_multiple', [True, False],
            ids=["gs_is_multiple", "gs_is_not_multiple"])
    if 'gl_size' in metafunc.funcargnames:
        grid_sizes = [
            (13,), (35,), (31*31*4,),
            (13, 15), (35, 13),
            (13, 15, 17), (75, 33, 5)]
        local_sizes = [None, (4,), (4, 4), (4, 4, 4)]
        gl_sizes = [(g, l) for g, l in itertools.product(grid_sizes, local_sizes)
            if l is None or len(g) == len(l)]
        metafunc.parametrize('gl_size', gl_sizes, ids=[str(x) for x in gl_sizes])
    if 'incorrect_gl_size' in metafunc.funcargnames:
        grid_sizes = [
            (31**3+1,),
            (31**2, 32), (31*20, 31*20),
            (31, 31, 32), (150, 150, 150)]
        local_sizes = [(4,), (4, 4), (4, 4, 4)]
        gl_sizes = [(g, l) for g, l in itertools.product(grid_sizes, local_sizes)
            if len(g) == len(l)]
        metafunc.parametrize('incorrect_gl_size', gl_sizes, ids=[str(x) for x in gl_sizes])


class ReferenceIds:

    def __init__(self, global_size, local_size):
        self.global_size = global_size
        self.np_global_size = tuple(reversed(global_size))
        if local_size is not None:
            self.local_size = local_size
            self.np_local_size = tuple(reversed(local_size))
            self.grid_size = tuple(min_blocks(gs, ls) for gs, ls in zip(global_size, local_size))
            self.np_grid_size = tuple(reversed(self.grid_size))

    def _tile_pattern(self, pattern, axis, full_shape):

        pattern_shape = [x if i == axis else 1 for i, x in enumerate(full_shape)]
        pattern = pattern.reshape(*pattern_shape)

        tile_shape = [x if i != axis else 1 for i, x in enumerate(full_shape)]
        pattern = numpy.tile(pattern, tile_shape)

        return pattern.astype(numpy.int32)

    def predict_local_ids(self, dim):
        np_dim = len(self.global_size) - dim - 1
        global_len = self.np_global_size[np_dim]
        local_len = self.np_local_size[np_dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.tile(numpy.arange(local_len), repetitions)[:global_len]
        return self._tile_pattern(pattern, np_dim, self.np_global_size)

    def predict_group_ids(self, dim):
        np_dim = len(self.global_size) - dim - 1
        global_len = self.np_global_size[np_dim]
        local_len = self.np_local_size[np_dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.repeat(numpy.arange(repetitions), local_len)[:global_len]
        return self._tile_pattern(pattern, np_dim, self.np_global_size)

    def predict_global_ids(self, dim):
        np_dim = len(self.global_size) - dim - 1
        global_len = self.np_global_size[np_dim]

        pattern = numpy.arange(global_len)
        return self._tile_pattern(pattern, np_dim, self.np_global_size)


class TestVirtualSizes:

    @classmethod
    def try_create(cls, global_size, local_size, max_num_groups, max_work_item_sizes):
        if local_size is not None and len(global_size) > len(local_size):
            local_size = local_size + (1,) * (len(global_size) - len(local_size))

        if local_size is None:
            max_global_size = max_num_groups
        else:
            if product(local_size) < product(max_work_item_sizes):
                return None

            max_global_size = [
                num_groups * ls for num_groups, ls
                in zip(max_num_groups, local_size)]

        if product(global_size) > product(max_global_size):
            return None

        return cls(global_size, local_size, max_num_groups, max_work_item_sizes)

    def __init__(self, global_size, local_size, max_num_groups, max_work_item_sizes):
        self.global_size = global_size
        self.local_size = local_size
        self.max_num_groups = max_num_groups
        self.max_work_item_sizes = max_work_item_sizes

    def is_supported_by(self, thr):
        return (
            len(self.max_num_groups) < len(thr.device_params.max_num_groups) and
            all(ng <= mng for ng, mng
                in zip(self.max_num_groups, thr.device_params.max_num_groups)) or
            len(self.max_work_item_sizes) > len(thr.device_params.max_work_item_sizes) or
            all(nwi <= mwi for nwi, mnwi
                in zip(self.max_work_item_sizes, thr.device_params.max_work_item_sizes)))

    def __str__(self):
        return "{gs}-{ls}-limited-by-{mng}-{mwis}".format(
            gs=self.global_size, ls=self.local_size,
            mng=self.max_num_groups, mwis=self.max_work_item_sizes)


def atest_ids(thr, testvs):
    """
    Test that virtual IDs are correct for each thread.
    """
    if not testvs.is_supported_by(thr):
        pytest.skip()

    thr.override_device_params(
        max_num_groups=testvs.max_num_groups,
        max_work_item_sizes=testvs.max_work_item_sizes)

    ref = ReferenceIds(testvs.global_size, testvs.local_size)

    get_ids = thr.compile_static("""
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

    print(get_ids.global_size, get_ids.local_size, get_ids.virtual_global_size, get_ids.virtual_local_size)

    local_ids = thr.array(ref.np_global_size, numpy.int32)
    group_ids = thr.array(ref.np_global_size, numpy.int32)
    global_ids = thr.array(ref.np_global_size, numpy.int32)

    for vdim in range(len(testvs.global_size)):

        get_ids(local_ids, group_ids, global_ids, numpy.int32(vdim))

        assert diff_is_negligible(global_ids.get(), ref.predict_global_ids(vdim))
        if local_size is not None:
            assert diff_is_negligible(local_ids.get(), ref.predict_local_ids(vdim))
            assert diff_is_negligible(group_ids.get(), ref.predict_group_ids(vdim))


def atest_sizes(thr_with_gs_limits, gl_size, gs_is_multiple):
    """
    Test that virtual sizes are correct.
    """

    thr = thr_with_gs_limits
    grid_size, local_size = gl_size
    if product(grid_size) > product(thr.device_params.max_num_groups):
        pytest.skip()

    global_size, ls = get_global_size(grid_size, local_size, gs_is_multiple=gs_is_multiple)
    ref = ReferenceIds(global_size, ls)

    get_sizes = thr.compile_static("""
    KERNEL void get_sizes(GLOBAL_MEM int *sizes)
    {
        if (virtual_global_id(0) > 0) return;

        for (int i = 0; i < 3; i++)
        {
            sizes[i] = virtual_local_size(i);
            sizes[i + 3] = virtual_num_groups(i);
            sizes[i + 6] = virtual_global_size(i);
        }
        sizes[9] = virtual_global_flat_size();
    }
    """, 'get_sizes', global_size, local_size=local_size)

    sizes = thr.array(10, numpy.int32)
    get_sizes(sizes)

    sizes = sizes.get()
    ls = sizes[0:3]
    gs = sizes[3:6]
    gls = sizes[6:9]
    flat_size = sizes[9]

    gls_ref = numpy.array(list(ref.global_size) + [1] * (3 - len(ref.global_size)), numpy.int32)
    assert diff_is_negligible(gls, gls_ref)
    assert flat_size == product(gls_ref)

    if local_size is not None:
        ls_ref = numpy.array(list(ref.local_size) + [1] * (3 - len(ref.local_size)), numpy.int32)
        gs_ref = numpy.array([min_blocks(g, l) for g, l in zip(gls_ref, ls_ref)], numpy.int32)
        assert diff_is_negligible(ls, ls_ref)
        assert diff_is_negligible(gs, gs_ref)


def atest_incorrect_sizes(thr_with_gs_limits, incorrect_gl_size):
    """
    Test that for sizes which exceed the thread capability, the exception is raised.
    """

    thr = thr_with_gs_limits
    grid_size, local_size = incorrect_gl_size

    global_size, ls = get_global_size(grid_size, local_size)
    ref = ReferenceIds(global_size, local_size)

    with pytest.raises(ValueError):
        kernel = thr.compile_static("""
        KERNEL void test(GLOBAL_MEM int *temp)
        {
            temp[0] = 1;
        }
        """, 'test', ref.global_size, local_size=ref.local_size)
