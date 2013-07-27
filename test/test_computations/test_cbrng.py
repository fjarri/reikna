import itertools
import numpy
import pytest

from helpers import *
from test_computations.cbrng_ref import philox, threefry

from reikna.helpers import product
from reikna.cbrng import CBRNG
from reikna.cbrng.rngs import Threefry, Philox
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


def raw_ref(counters, keys, name, words, bitness, rounds):
    result = numpy.empty_like(counters)
    func = philox if name == 'philox' else threefry

    for i in range(keys.shape[0]):
        result[i] = func(bitness, words, counters[i], keys[i], Nrounds=rounds)

    return result


def test_raw(thr, name_and_params):
    name, params = name_and_params

    size = 1000
    key = 123
    words = params['words']
    bitness = params['bitness']
    rounds = params['rounds']

    if name == 'philox':
        rng_ctr = Philox
    else:
        rng_ctr = Threefry
    rng = rng_ctr(bitness, words, rounds=rounds)

    keys_ref = numpy.zeros((size, rng.key_words), rng.dtype)
    key0 = key << (32 if rng.dtype.itemsize == 8 and rng.key_words == 1 else 0)
    keys_ref[:,0] = key0
    keys_ref[:,rng.key_words-1] += numpy.arange(size)
    counters_ref = numpy.zeros((size, rng.counter_words), rng.dtype)

    rng_kernel = thr.compile_static(
        """
        <%
            ctype = dtypes.ctype(rng.dtype)
        %>
        KERNEL void test(GLOBAL_MEM ${ctype} *dest, int ctr)
        {
            VIRTUAL_SKIP_THREADS;
            const int idx = virtual_global_id(0);

            ${rng.module}KEY key;
            key.v[0] = ${key0};
            %for i in range(1, rng.key_words):
            key.v[${i}] = 0;
            %endfor
            key.v[${rng.key_words - 1}] += idx;

            ${rng.module}COUNTER counter;
            %for i in range(rng.counter_words - 1):
            counter.v[${i}] = 0;
            %endfor
            counter.v[${rng.counter_words - 1}] = ctr;

            ${rng.module}COUNTER result = ${rng.module}(key, counter);

            %for i in range(rng.counter_words):
            dest[idx * ${rng.counter_words} + ${i}] = result.v[${i}];
            %endfor
        }
        """,
        'test', size,
        render_kwds=dict(rng=rng, key0=key0))

    dest = thr.array((size, rng.counter_words), rng.dtype)

    rng_kernel(dest, numpy.int32(0))
    dest_ref = raw_ref(counters_ref, keys_ref, name, words, bitness, rounds)
    assert diff_is_negligible(dest.get(), dest_ref)

    rng_kernel(dest, numpy.int32(1))
    counters_ref[:,-1] += 1
    dest_ref = raw_ref(counters_ref, keys_ref, name, words, bitness, rounds)
    assert diff_is_negligible(dest.get(), dest_ref)


def check_distribution(thr, rng_name, rng_params,
        distribution, distribution_params, dtype, reference):

    size = 10000
    batch = 100
    seed = 456

    dest_dev = thr.array((batch, size), dtype)
    rng = CBRNG(dest_dev, 1, seed=seed, rng=rng_name, rng_params=rng_params,
        distribution=distribution, distribution_params=distribution_params)
    rngc = rng.compile(thr)

    counters = create_counters(size, rng_params)
    counters_dev = thr.to_device(counters)

    rngc(counters_dev, dest_dev, counters_dev)
    dest = dest_dev.get()

    extent = reference.get('extent', None)
    mean = reference.get('mean', None)
    std = reference.get('std', None)

    if extent is not None:
        assert dest.min() >= extent[0]
        assert dest.max() <= extent[1]

    if mean is not None and std is not None:
        # expected mean and std of the mean of the sample array
        m_mean = mean
        m_std = std / numpy.sqrt(batch * size)

        diff = abs(dest.mean() - mean)
        assert diff < 5 * m_std # about 1e-6 chance of fail

    if std is not None:
        # expected mean and std of the variance of the sample array
        v_mean = std ** 2
        v_std = numpy.sqrt(2. * std ** 4 / (batch * size - 1))

        diff = abs(dest.var() - v_mean)
        assert diff < 5 * v_std # about 1e-6 chance of fail


def uniform_discrete_mean_and_std(min, max):
    return (min + max) / 2., numpy.sqrt(((max - min + 1) ** 2 - 1.) / 12)


def uniform_mean_and_std(min, max):
    return (min + max) / 2., (max - min) / numpy.sqrt(12)


def test_32_to_64_bit(thr):
    extent = (0, 2**63-1)
    mean, std = uniform_discrete_mean_and_std(*extent)
    check_distribution(thr,
        'philox', dict(bitness=32, words=4),
        'uniform_integer', dict(min=extent[0], max=extent[1] + 1), numpy.uint64,
        dict(extent=extent, mean=mean, std=std))


def test_64_to_32_bit(thr):
    extent = (0, 2**31-1)
    mean, std = uniform_discrete_mean_and_std(*extent)
    check_distribution(thr,
        'philox', dict(bitness=64, words=4),
        'uniform_integer', dict(min=extent[0], max=extent[1] + 1), numpy.uint32,
        dict(extent=extent, mean=mean, std=std))


def test_uniform_integer(thr):
    extent = (-10, 98)
    mean, std = uniform_discrete_mean_and_std(*extent)
    check_distribution(thr,
        'philox', dict(bitness=64, words=4),
        'uniform_integer', dict(min=extent[0], max=extent[1] + 1), numpy.int32,
        dict(extent=extent, mean=mean, std=std))


def test_uniform_float(thr_and_double):
    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32
    extent = (-5, 7.7)
    mean, std = uniform_mean_and_std(*extent)
    check_distribution(thr,
        'philox', dict(bitness=64, words=4),
        'uniform_float', dict(min=extent[0], max=extent[1]), dtype,
        dict(extent=extent, mean=mean, std=std))


def test_normal_bm(thr_and_double):
    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32
    mean, std = -2, 10
    check_distribution(thr,
        'philox', dict(bitness=64, words=4),
        'normal_bm', dict(mean=mean, std=std), dtype,
        dict(mean=mean, std=std))


def test_gamma(thr_and_double):
    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32
    shape, scale = 3, 10
    mean = shape * scale
    std = numpy.sqrt(shape) * scale
    check_distribution(thr,
        'philox', dict(bitness=64, words=4),
        'gamma', dict(mean=mean, std=std), dtype,
        dict(shape=shape, scale=scale))
