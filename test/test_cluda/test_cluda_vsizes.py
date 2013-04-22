import itertools

import pytest

import reikna.cluda as cluda
from reikna.helpers import min_blocks, product
import reikna.cluda.dtypes as dtypes
from helpers import *

from pytest_threadgen import parametrize_thread_tuple, create_thread_in_tuple


pytest_funcarg__thr_with_gs_limits = create_thread_in_tuple


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
        if len(gl) > len(mgs) or (len(mgs) > 2 and len(gl) > 2 and mgs[2] < gl[2]):
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
        local_sizes = [(4,), (4, 4), (4, 4, 4)]
        gl_sizes = [(g, l) for g, l in itertools.product(grid_sizes, local_sizes)
            if len(g) == len(l)]
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

    def __init__(self, grid_size, local_size, gs_is_multiple=True):
        global_size = [g * l for g, l in zip(grid_size, local_size)]
        if not gs_is_multiple:
            global_size = [g - 1 for g in global_size]

        self.global_size = tuple(global_size)
        self.local_size = local_size
        self.np_global_size = list(reversed(global_size))
        self.np_local_size = list(reversed(local_size))

    def predict_global_flat_ids(self):
        return numpy.arange(product(self.np_global_size)).astype(numpy.int32)

    def predict_local_ids(self, dim):
        if dim > len(self.global_size) - 1:
            return numpy.zeros(self.np_global_size, dtype=numpy.int32)

        np_dim = len(self.global_size) - dim - 1

        global_len = self.np_global_size[np_dim]
        local_len = self.np_local_size[np_dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.tile(numpy.arange(local_len), repetitions)[:global_len]

        pattern_shape = [x if i == np_dim else 1 for i, x in enumerate(self.np_global_size)]
        pattern = pattern.reshape(*pattern_shape)

        tile_shape = [x if i != np_dim else 1 for i, x in enumerate(self.np_global_size)]
        pattern = numpy.tile(pattern, tile_shape)

        return pattern.astype(numpy.int32)

    def predict_group_ids(self, dim):
        if dim > len(self.global_size) - 1:
            return numpy.zeros(self.np_global_size, dtype=numpy.int32)

        np_dim = len(self.global_size) - dim - 1

        global_len = self.np_global_size[np_dim]
        local_len = self.np_local_size[np_dim]
        repetitions = min_blocks(global_len, local_len)

        pattern = numpy.repeat(numpy.arange(repetitions), local_len)[:global_len]

        pattern_shape = [x if i == np_dim else 1 for i, x in enumerate(self.np_global_size)]
        pattern = pattern.reshape(*pattern_shape)

        tile_shape = [x if i != np_dim else 1 for i, x in enumerate(self.np_global_size)]
        pattern = numpy.tile(pattern, tile_shape)

        return pattern.astype(numpy.int32)

    def predict_global_ids(self, dim):
        lids = self.predict_local_ids(dim)
        gids = self.predict_group_ids(dim)
        return lids + gids * (self.local_size[dim] if dim < len(self.local_size) else 0)


def test_ids(thr_with_gs_limits, gl_size, gs_is_multiple):
    """
    Test that virtual IDs are correct for each thread.
    """

    thr = thr_with_gs_limits
    grid_size, local_size = gl_size
    if product(grid_size) > product(thr.device_params.max_num_groups):
        pytest.skip()

    ref = ReferenceIds(grid_size, local_size, gs_is_multiple)

    get_ids = thr.compile_static("""
    KERNEL void get_ids(GLOBAL_MEM int *fid,
        GLOBAL_MEM int *lx, GLOBAL_MEM int *ly, GLOBAL_MEM int *lz,
        GLOBAL_MEM int *gx, GLOBAL_MEM int *gy, GLOBAL_MEM int *gz,
        GLOBAL_MEM int *glx, GLOBAL_MEM int *gly, GLOBAL_MEM int *glz)
    {
        VIRTUAL_SKIP_THREADS;
        const int i = virtual_global_flat_id();
        fid[i] = i;
        lx[i] = virtual_local_id(0);
        ly[i] = virtual_local_id(1);
        lz[i] = virtual_local_id(2);
        gx[i] = virtual_group_id(0);
        gy[i] = virtual_group_id(1);
        gz[i] = virtual_group_id(2);
        glx[i] = virtual_global_id(0);
        gly[i] = virtual_global_id(1);
        glz[i] = virtual_global_id(2);
    }
    """, 'get_ids', ref.global_size, local_size=ref.local_size)

    fid = thr.array(product(ref.np_global_size), numpy.int32)
    lx = thr.array(ref.np_global_size, numpy.int32)
    ly = thr.array(ref.np_global_size, numpy.int32)
    lz = thr.array(ref.np_global_size, numpy.int32)
    gx = thr.array(ref.np_global_size, numpy.int32)
    gy = thr.array(ref.np_global_size, numpy.int32)
    gz = thr.array(ref.np_global_size, numpy.int32)
    glx = thr.array(ref.np_global_size, numpy.int32)
    gly = thr.array(ref.np_global_size, numpy.int32)
    glz = thr.array(ref.np_global_size, numpy.int32)

    get_ids(fid, lx, ly, lz, gx, gy, gz, glx, gly, glz)

    assert diff_is_negligible(fid.get(), ref.predict_global_flat_ids())
    assert diff_is_negligible(lx.get(), ref.predict_local_ids(0))
    assert diff_is_negligible(ly.get(), ref.predict_local_ids(1))
    assert diff_is_negligible(lz.get(), ref.predict_local_ids(2))
    assert diff_is_negligible(gx.get(), ref.predict_group_ids(0))
    assert diff_is_negligible(gy.get(), ref.predict_group_ids(1))
    assert diff_is_negligible(gz.get(), ref.predict_group_ids(2))
    assert diff_is_negligible(glx.get(), ref.predict_global_ids(0))
    assert diff_is_negligible(gly.get(), ref.predict_global_ids(1))
    assert diff_is_negligible(glz.get(), ref.predict_global_ids(2))


def test_sizes(thr_with_gs_limits, gl_size, gs_is_multiple):
    """
    Test that virtual sizes are correct.
    """

    thr = thr_with_gs_limits
    grid_size, local_size = gl_size

    ref = ReferenceIds(grid_size, local_size, gs_is_multiple)

    get_sizes = thr.compile_static("""
    KERNEL void get_sizes(GLOBAL_MEM int *sizes)
    {
        if (virtual_global_flat_id() > 0) return;

        for (int i = 0; i < 3; i++)
        {
            sizes[i] = virtual_local_size(i);
            sizes[i + 3] = virtual_num_groups(i);
            sizes[i + 6] = virtual_global_size(i);
        }
        sizes[9] = virtual_global_flat_size();
    }
    """, 'get_sizes', ref.global_size, local_size=ref.local_size)

    sizes = thr.array(10, numpy.int32)
    get_sizes(sizes)

    gls = list(ref.global_size) + [1] * (3 - len(ref.global_size))
    ls = list(ref.local_size) + [1] * (3 - len(ref.local_size))
    gs = [min_blocks(g, l) for g, l in zip(gls, ls)]

    ref_sizes = numpy.array(ls + gs + gls + [product(gls)]).astype(numpy.int32)
    assert diff_is_negligible(sizes.get(), ref_sizes)


def test_incorrect_sizes(thr_with_gs_limits, incorrect_gl_size):
    """
    Test that for sizes which exceed thread capability the exception is raised
    """

    thr = thr_with_gs_limits
    grid_size, local_size = incorrect_gl_size

    ref = ReferenceIds(grid_size, local_size)

    with pytest.raises(ValueError):
        kernel = thr.compile_static("""
        KERNEL void test(GLOBAL_MEM int *temp)
        {
            temp[0] = 1;
        }
        """, 'test', ref.global_size, local_size=ref.local_size)
