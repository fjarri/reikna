import itertools
import time

import pytest
import numpy
from scipy.special import iv

from helpers import *
from .cbrng_ref import philox as philox_ref
from .cbrng_ref import threefry as threefry_ref

from reikna.core import Type
from reikna.helpers import product
from reikna.cbrng import CBRNG
from reikna.cbrng.bijections import threefry, philox
from reikna.cbrng.tools import KeyGenerator
from reikna.cbrng.samplers import uniform_integer, uniform_float, normal_bm, gamma, vonmises
import reikna.cluda.dtypes as dtypes


def uniform_discrete_mean_and_std(min, max):
    return (min + max) / 2., numpy.sqrt(((max - min + 1) ** 2 - 1.) / 12)


def uniform_mean_and_std(min, max):
    return (min + max) / 2., (max - min) / numpy.sqrt(12)


class UniformIntegerHelper:
    def __init__(self, min_, max_):
        self.extent = (min_, max_)
        self.mean, self.std = uniform_discrete_mean_and_std(*self.extent)
        self.name = 'uniform_integer'

    def get_sampler(self, bijection, dtype):
        return uniform_integer(bijection, dtype, self.extent[0], self.extent[1] + 1)

class UniformFloatHelper:
    def __init__(self, min_, max_):
        self.extent = (min_, max_)
        self.mean, self.std = uniform_mean_and_std(*self.extent)
        self.name = 'uniform_float'

    def get_sampler(self, bijection, double):
        dtype = numpy.float64 if double else numpy.float32
        return uniform_float(bijection, dtype, self.extent[0], self.extent[1])

class NormalBMHelper:
    def __init__(self, mean, std):
        self.extent = None
        self.mean = mean
        self.std = std
        self.name = 'normal_bm'

    def get_sampler(self, bijection, double):
        dtype = numpy.float64 if double else numpy.float32
        return normal_bm(bijection, dtype, mean=self.mean, std=self.std)

class NormalBMComplexHelper:
    def __init__(self, mean, std):
        self.extent = None
        self.mean = mean
        self.std = std
        self.name = 'normal_bm_complex'

    def get_sampler(self, bijection, double):
        dtype = numpy.complex128 if double else numpy.complex64
        return normal_bm(bijection, dtype, mean=self.mean, std=self.std)

class GammaHelper:
    def __init__(self, shape, scale):
        self._shape = shape
        self._scale = scale
        self.mean = shape * scale
        self.std = numpy.sqrt(shape) * scale
        self.extent = None
        self.name = 'gamma'

    def get_sampler(self, bijection, double):
        dtype = numpy.float64 if double else numpy.float32
        return gamma(bijection, dtype, shape=self._shape, scale=self._scale)


class VonMisesHelper:
    def __init__(self, mu, kappa):
        self._mu = mu
        self._kappa = kappa
        self.circular_mean = mu
        self.circular_var = 1 - iv(1, kappa) / iv(0, kappa)
        self.extent = (-numpy.pi, numpy.pi)
        self.name = 'vonmises'

    def get_sampler(self, bijection, double):
        dtype = numpy.float64 if double else numpy.float32
        return vonmises(bijection, dtype, mu=self._mu, kappa=self._kappa)


class BijectionHelper:

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
            val = BijectionHelper(name, words, bitness)
            vals.append(val)
            ids.append(str(val))

        metafunc.parametrize('test_bijection', vals, ids=ids)

    if 'test_sampler_int' in metafunc.funcargnames:
        vals = [
            UniformIntegerHelper(-10, 98)]
        ids = [test.name for test in vals]
        metafunc.parametrize('test_sampler_int', vals, ids=ids)

    if 'test_sampler_float' in metafunc.funcargnames:
        vals = [
            UniformFloatHelper(-5, 7.7),
            NormalBMHelper(-2, 10),
            NormalBMComplexHelper(-3 + 4j, 7),
            GammaHelper(3, 10),
            VonMisesHelper(1, 0.7),
            ]

        ids = [test.name for test in vals]
        metafunc.parametrize('test_sampler_float', vals, ids=ids)


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


def check_kernel_sampler(thr, sampler, ref):

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
        }
        """,
        'test', size,
        render_kwds=dict(
            size=size, batch=batch, ctype=dtypes.ctype(sampler.dtype),
            bijection=bijection, keygen=keygen, sampler=sampler))

    dest = thr.array((batch, sampler.randoms_per_call, size), sampler.dtype)
    rng_kernel(dest, numpy.int32(0))
    dest = dest.get()

    check_distribution(dest, ref)


def check_distribution(arr, ref):
    extent = getattr(ref, 'extent', None)
    mean = getattr(ref, 'mean', None)
    std = getattr(ref, 'std', None)
    circular_mean = getattr(ref, 'circular_mean', None)
    circular_var = getattr(ref, 'circular_var', None)

    if extent is not None:
        assert arr.min() >= extent[0]
        assert arr.max() <= extent[1]

    if circular_mean is not None and circular_var is not None:
        z = numpy.exp(1j * arr)
        arr_cmean = numpy.angle(z.mean())
        arr_R = numpy.abs(z.mean())
        arr_cvar = 1 - arr_R

        # FIXME: need a valid mathematical formula for the standard error of the mean
        # for circular distributions.
        # Currently it is just a rough estimate.
        m_std = circular_var**0.5 / numpy.sqrt(arr.size)
        diff = abs(arr_cmean - circular_mean)
        assert diff < 5 * m_std

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


def test_32_to_64_bit(thr):
    bijection = philox(32, 4)
    ref = UniformIntegerHelper(0, 2**63-1)
    sampler = ref.get_sampler(bijection, numpy.uint64)
    check_kernel_sampler(thr, sampler, ref)


def test_64_to_32_bit(thr):
    bijection = philox(64, 4)
    ref = UniformIntegerHelper(0, 2**31-1)
    sampler = ref.get_sampler(bijection, numpy.uint32)
    check_kernel_sampler(thr, sampler, ref)


def test_kernel_sampler_int(thr, test_sampler_int):
    bijection = philox(64, 4)
    check_kernel_sampler(
        thr, test_sampler_int.get_sampler(bijection, numpy.int32), test_sampler_int)

def test_kernel_sampler_float(thr_and_double, test_sampler_float):
    thr, double = thr_and_double
    bijection = philox(64, 4)
    check_kernel_sampler(
        thr, test_sampler_float.get_sampler(bijection, double), test_sampler_float)


def check_computation(thr, rng, ref):
    dest_dev = thr.empty_like(rng.parameter.randoms)
    counters = rng.create_counters()
    counters_dev = thr.to_device(counters)
    rngc = rng.compile(thr)

    rngc(counters_dev, dest_dev)
    dest = dest_dev.get()
    check_distribution(dest, ref)


def test_computation_general(thr_and_double):

    size = 10000
    batch = 101

    thr, double = thr_and_double

    bijection = philox(64, 4)
    ref = NormalBMHelper(mean=-2, std=10)
    sampler = ref.get_sampler(bijection, double)

    rng = CBRNG(Type(sampler.dtype, shape=(batch, size)), 1, sampler)
    check_computation(thr, rng, ref)


def test_computation_convenience(thr):

    size = 10000
    batch = 101

    ref = UniformIntegerHelper(0, 511)
    rng = CBRNG.uniform_integer(Type(numpy.int32, shape=(batch, size)), 1,
        sampler_kwds=dict(low=ref.extent[0], high=ref.extent[1]))
    check_computation(thr, rng, ref)


def test_computation_uniqueness(thr):
    """
    A regression test for the bug with a non-updating counter.
    """

    size = 10000
    batch = 1

    rng = CBRNG.normal_bm(Type(numpy.complex64, shape=(batch, size)), 1)

    dest1_dev = thr.empty_like(rng.parameter.randoms)
    dest2_dev = thr.empty_like(rng.parameter.randoms)
    counters = rng.create_counters()
    counters_dev = thr.to_device(counters)
    rngc = rng.compile(thr)

    rngc(counters_dev, dest1_dev)
    rngc(counters_dev, dest2_dev)

    assert not diff_is_negligible(dest1_dev.get(), dest2_dev.get(), verbose=False)


@pytest.mark.perf
@pytest.mark.returns('GB/s')
def test_computation_performance(thr_and_double, fast_math, test_sampler_float):

    thr, double = thr_and_double

    size = 2 ** 15
    batch = 2 ** 6

    bijection = philox(64, 4)
    sampler = test_sampler_float.get_sampler(bijection, double)

    rng = CBRNG(Type(sampler.dtype, shape=(batch, size)), 1, sampler)

    dest_dev = thr.empty_like(rng.parameter.randoms)
    counters = rng.create_counters()
    counters_dev = thr.to_device(counters)
    rngc = rng.compile(thr, fast_math=fast_math)

    attempts = 10
    times = []
    for i in range(attempts):
        t1 = time.time()
        rngc(counters_dev, dest_dev)
        thr.synchronize()
        times.append(time.time() - t1)

    byte_size = size * batch * sampler.dtype.itemsize
    return min(times), byte_size
