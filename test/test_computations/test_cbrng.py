import itertools
import numpy
import pytest

from helpers import *
from test_computations.cbrng_ref import philox as philox_ref
from test_computations.cbrng_ref import threefry as threefry_ref

from reikna.core import Type
from reikna.helpers import product
from reikna.cbrng import CBRNG
from reikna.cbrng.bijections import threefry, philox
from reikna.cbrng.tools import KeyGenerator
from reikna.cbrng.samplers import uniform_integer, uniform_float, normal_bm, gamma
import reikna.cluda.dtypes as dtypes


class TestBijection:

    def __init__(self, name, words, bitness):
        rounds = 20 if name == 'threefry' else 10
        if name == 'philox':
            bijection_func = philox
        else:
            bijection_func = threefry
        self._name = name
        self._words = words
        self._bitness = bitness
        self._rounds = rounds
        self.bijection = bijection_func(bitness, words, rounds=rounds)

        func = philox_ref if name == 'philox' else threefry_ref
        self._reference_func = lambda ctr, key: func(bitness, words, ctr, key, Nrounds=rounds)

    def reference(self, counters, keygen):
        result = numpy.empty_like(counters)
        for i in range(counters.shape[0]):
            result[i] = self._reference_func(counters[i]['v'], keygen(i)['v'])
        return result

    def __str__(self):
        return "{name}-{words}x{bitness}-{rounds}".format(
            name=self._name, words=self._words, bitness=self._bitness, rounds=self._rounds)


def pytest_generate_tests(metafunc):

    if 'test_bijection' in metafunc.funcargnames:

        vals = []
        ids = []

        for name, words, bitness in itertools.product(['threefry', 'philox'], [2, 4], [32, 64]):
            val = TestBijection(name, words, bitness)
            vals.append(val)
            ids.append(str(val))

        metafunc.parametrize('test_bijection', vals, ids=ids)


def test_kernel_bijection(thr, test_bijection):

    size = 1000
    seed = 123

    bijection = test_bijection.bijection
    keygen = KeyGenerator.create(bijection, seed=seed, reserve_id_space=False)
    counters_ref = numpy.zeros(size, bijection.counter_dtype)

    rng_kernel = thr.compile_static(
        """
        KERNEL void test(GLOBAL_MEM ${bijection.module}Counter *dest, int ctr)
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T idx = virtual_global_id(0);

            ${bijection.module}Key key = ${keygen.module}key_from_int(idx);
            ${bijection.module}Counter counter = ${bijection.module}make_counter_from_int(ctr);
            ${bijection.module}Counter result = ${bijection.module}bijection(key, counter);

            dest[idx] = result;
        }
        """,
        'test', size,
        render_kwds=dict(bijection=bijection, keygen=keygen))

    dest = thr.array(size, bijection.counter_dtype)

    rng_kernel(dest, numpy.int32(0))
    dest_ref = test_bijection.reference(counters_ref, keygen.reference)
    assert (dest.get() == dest_ref).all()

    rng_kernel(dest, numpy.int32(1))
    counters_ref['v'][:,-1] = 1
    dest_ref = test_bijection.reference(counters_ref, keygen.reference)
    assert (dest.get() == dest_ref).all()


def check_kernel_sampler(thr, sampler, extent=None, mean=None, std=None):

    size = 10000
    batch = 100
    seed = 456

    bijection = sampler.bijection
    keygen = KeyGenerator.create(bijection, seed=seed)

    rng_kernel = thr.compile_static(
        """
        KERNEL void test(GLOBAL_MEM ${ctype} *dest, int ctr_start)
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T idx = virtual_global_id(0);

            ${bijection.module}Key key = ${keygen.module}key_from_int(idx);
            ${bijection.module}Counter ctr = ${bijection.module}make_counter_from_int(ctr_start);
            ${bijection.module}State st = ${bijection.module}make_state(key, ctr);

            ${sampler.module}Result res;

            for(int j = 0; j < ${batch}; j++)
            {
                res = ${sampler.module}sample(&st);

                %for i in range(sampler.randoms_per_call):
                dest[j * ${size * sampler.randoms_per_call} + ${size * i} + idx] = res.v[${i}];
                %endfor
            }

            ${bijection.module}Counter next_ctr = ${bijection.module}get_next_unused_counter(st);
        }
        """,
        'test', size,
        render_kwds=dict(
            size=size, batch=batch, ctype=dtypes.ctype(sampler.dtype),
            bijection=bijection, keygen=keygen, sampler=sampler))

    dest = thr.array((batch, sampler.randoms_per_call, size), sampler.dtype)
    rng_kernel(dest, numpy.int32(0))
    dest = dest.get()

    check_distribution(dest, extent=extent, mean=mean, std=std)


def check_distribution(arr, extent=None, mean=None, std=None):
    if extent is not None:
        assert arr.min() >= extent[0]
        assert arr.max() <= extent[1]

    if mean is not None and std is not None:
        # expected mean and std of the mean of the sample array
        m_mean = mean
        m_std = std / numpy.sqrt(arr.size)

        diff = abs(arr.mean() - mean)
        assert diff < 5 * m_std # about 1e-6 chance of fail

    if std is not None:
        # expected mean and std of the variance of the sample array
        v_mean = std ** 2
        v_std = numpy.sqrt(2. * std ** 4 / (arr.size - 1))

        diff = abs(arr.var() - v_mean)
        assert diff < 5 * v_std # about 1e-6 chance of fail


def uniform_discrete_mean_and_std(min, max):
    return (min + max) / 2., numpy.sqrt(((max - min + 1) ** 2 - 1.) / 12)


def uniform_mean_and_std(min, max):
    return (min + max) / 2., (max - min) / numpy.sqrt(12)


def test_32_to_64_bit(thr):
    extent = (0, 2**63-1)
    mean, std = uniform_discrete_mean_and_std(*extent)
    bijection = philox(32, 4)
    sampler = uniform_integer(bijection, numpy.uint64, extent[0], extent[1] + 1)
    check_kernel_sampler(thr, sampler, extent=extent, mean=mean, std=std)


def test_64_to_32_bit(thr):
    extent = (0, 2**31-1)
    mean, std = uniform_discrete_mean_and_std(*extent)
    bijection = philox(64, 4)
    sampler = uniform_integer(bijection, numpy.uint32, extent[0], extent[1] + 1)
    check_kernel_sampler(thr, sampler, extent=extent, mean=mean, std=std)


def test_uniform_integer(thr):
    extent = (-10, 98)
    mean, std = uniform_discrete_mean_and_std(*extent)
    bijection = philox(64, 4)
    sampler = uniform_integer(bijection, numpy.int32, extent[0], extent[1] + 1)
    check_kernel_sampler(thr, sampler, extent=extent, mean=mean, std=std)


def test_uniform_float(thr_and_double):
    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32
    extent = (-5, 7.7)
    mean, std = uniform_mean_and_std(*extent)
    bijection = philox(64, 4)
    sampler = uniform_float(bijection, dtype, extent[0], extent[1])
    check_kernel_sampler(thr, sampler, extent=extent, mean=mean, std=std)


def test_normal_bm(thr_and_double):
    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32
    mean, std = -2, 10
    bijection = philox(64, 4)
    sampler = normal_bm(bijection, dtype, mean=mean, std=std)
    check_kernel_sampler(thr, sampler, mean=mean, std=std)


def test_gamma(thr_and_double):
    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32
    shape, scale = 3, 10
    mean = shape * scale
    std = numpy.sqrt(shape) * scale
    bijection = philox(64, 4)
    sampler = gamma(bijection, dtype, shape=shape, scale=scale)
    check_kernel_sampler(thr, sampler, mean=mean, std=std)


def check_computation(thr, rng, extent=None, mean=None, std=None):
    dest_dev = thr.empty_like(rng.parameter.randoms)
    counters = rng.create_counters()
    counters_dev = thr.to_device(counters)
    rngc = rng.compile(thr)

    rngc(counters_dev, dest_dev)
    dest = dest_dev.get()
    check_distribution(dest, extent=extent, mean=mean, std=std)


def test_computation_general(thr_and_double):

    size = 10000
    batch = 101

    thr, double = thr_and_double
    dtype = numpy.float64 if double else numpy.float32
    mean, std = -2, 10
    bijection = philox(64, 4)
    sampler = normal_bm(bijection, dtype, mean=mean, std=std)

    rng = CBRNG(Type(dtype, shape=(batch, size)), 1, sampler)
    check_computation(thr, rng, mean=mean, std=std)


def test_computation_convenience(thr):

    size = 10000
    batch = 101

    extent = (0, 511)
    mean, std = uniform_discrete_mean_and_std(*extent)
    rng = CBRNG.uniform_integer(Type(numpy.int32, shape=(batch, size)), 1,
        sampler_kwds=dict(low=extent[0], high=extent[1] + 1))
    check_computation(thr, rng, extent=extent, mean=mean, std=std)
