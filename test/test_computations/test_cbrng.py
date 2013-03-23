import itertools
import numpy
import pytest

from helpers import *
from test_computations.cbrng_ref import philox, threefry

from reikna.helpers import product
from reikna.cbrng import CBRNG, create_counters, create_key
import reikna.cluda.dtypes as dtypes


def pytest_generate_tests(metafunc):

    if 'name_and_params' in metafunc.funcargnames:

        vals = []
        ids = []

        for name, words, bitness in itertools.product(['threefry', 'philox'], [2, 4], [32, 64]):
            # This particular set is not supported by CBRNG,
            # because the key in this case is only 32bit and cannot hold
            # both seed and thread id.
            if name == 'philox' and words == 2 and bitness == 32:
                continue

            rounds = 20 if name == 'threefry' else 10

            params = dict(words=words, bitness=bitness, rounds=rounds)
            vals.append((name, params))
            ids.append("{name}-{words}x{bitness}-{rounds}".format(name=name, **params))

        metafunc.parametrize('name_and_params', vals, ids=ids)


def rng_ref(ctr, size, name, seed, **params):
    dtype = numpy.uint32 if params['bitness'] == 32 else numpy.uint64
    result = numpy.empty((size, params['words']), dtype)
    base_key = create_key(name, params, seed=seed)

    func = philox if name == 'philox' else threefry

    counter = numpy.zeros(params['words'], dtype)
    counter[-1] += ctr

    for i in range(size):
        key = base_key.copy()
        key[-1] += numpy.cast[key.dtype](i)

        result[i] = func(params['bitness'], params['words'], counter, key, params['rounds'])

    return result.T


def test_raw(ctx, name_and_params):
    name, params = name_and_params
    distribution = 'uniform_integer'
    size = 1000
    seed = 123

    counters = create_counters(ctx, size, name, distribution, params)
    dest = ctx.array((params['words'], size),
        numpy.uint32 if params['bitness'] == 32 else numpy.uint64)

    rng = CBRNG(ctx).prepare_for(counters, dest, counters,
        seed=seed, rng=name, rng_params=params, distribution=distribution)

    rng(counters, dest, counters)
    dest_ref = rng_ref(0, size, name, seed, **params)
    assert diff_is_negligible(dest.get(), dest_ref)

    rng(counters, dest, counters)
    dest_ref = rng_ref(1, size, name, seed, **params)
    assert diff_is_negligible(dest.get(), dest_ref)


def test_uniform_float():
    pass


def test_gauss_bm():
    pass


def test_lambda():
    pass


@pytest.mark.perf
def test_raw_perf():
    pass
