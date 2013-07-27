import numpy

import reikna.helpers as helpers
import reikna.cluda.dtypes as dtypes
from reikna.cluda import Module

TEMPLATE = helpers.template_for(__file__)


class Distribution:
    """
    The base class for the kernel API of random distributions.
    Supports ``__process_modules__`` protocol.

    .. py:attribute:: deterministic

        If ``True``, every sampled random number consumes the same amount of counters.

    .. py:attribute:: module

        The module containing the distribution sampling function.
        Provides:
    """
    pass





class UniformInteger:
    """
    Generates uniformly distributed integer numbers in the interval ``[low, high)``.
    If ``high`` is ``None``, the interval is ``[0, low)``.
    Supported dtypes: ``(u)int(32/64)``.
    If the size of the interval is a power of 2, a fixed number of counters is used in each thread.
    """

    def __init__(self, cbrng, dtype, low, high=None, processed_module=None):
        if high is None:
            low, high = 0, low + 1
        else:
            assert low < high - 1

        self.randoms_per_call = 1

        num = high - low
        assert num <= 2 ** dtype.itemsize

        dtype = dtypes.normalize_type(dtype)
        ctype = dtypes.ctype(dtype)

        if num <= 2 ** 32:
            raw_ctype = dtypes.ctype(numpy.uint32)
            raw_func = 'get_raw_uint32'
            max_num = 2 ** 32
        else:
            raw_ctype = dtypes.ctype(numpy.uint64)
            raw_func = 'get_raw_uint64'
            max_num = 2 ** 64

        if processed_module is None:
            self.module = Module(
                TEMPLATE.get_def("uniform_integer"),
                render_kwds=dict(
                    cbrng=cbrng,
                    dtype=dtype, raw_ctype=raw_ctype, raw_func=raw_func,
                    max_num=max_num, num=num, low=low))
        else:
            self.module = processed_module


class UniformFloat:

    def __init__(self, dtype, low=0, high=1):
        """
        Generates uniformly distributed floating-points numbers in the interval ``[low, high)``.
        Supported dtypes: ``float(32/64)``.
        A fixed number of counters is used in each thread.
        """
        assert low < high
        self.randoms_per_call = 1

        ctype = dtypes.ctype(dtype)
        bitness = 64 if dtypes.is_double(dtype) else 32
        raw_func = 'get_raw_uint' + str(bitness)
        raw_max = dtypes.c_constant(2 ** bitness, dtype)

        size = dtypes.c_constant(high - low, dtype)
        low = dtypes.c_constant(low, dtype)

        self.module = Module(
            TEMPLATE.get_def("uniform_float"),
            render_kwds=dict(
                ctype=ctype,
                raw_func=raw_func, raw_max=raw_max, size=size, low=low))


class NormalBM:
    """
    Generates normally distributed random numbers with the mean ``mean`` and
    the standard deviation ``std`` using Box-Muller transform.
    Supported dtypes: ``float(32/64)``.
    A fixed number of counters is used in each thread.
    """

    def __init__(self, dtype, mean=0, std=1):
        self.randoms_per_call = 2

        dtype2 = dtypes.complex_for(dtype)
        ctype = dtypes.ctype(dtype)
        ctype2 = dtypes.ctype(dtype2)
        uniform_float = UniformFloat(0, 1).get_module(dtype)

        self.module = Module(
            TEMPLATE.get_def("normal_bm"),
            render_kwds=dict(
                ctype=ctype, ctype2=ctype2,
                mean=dtypes.c_constant(mean, dtype),
                std=dtypes.c_constant(std, dtype),
                uniform_float=uniform_float))


class Gamma:
    """
    Generates random numbers from the gamma distribution

    .. math::
      P(x) = x^{k-1} \\frac{e^{-x/\\theta}}{\\theta^k \\Gamma(k)},

    where :math:`k` is ``shape``, and :math:`\\theta` is ``scale``.
    Supported dtypes: ``float(32/64)``.
    """

    def __init__(self, shape=1, scale=1):
        self.randoms_per_call = 1

        dtype2 = dtypes.complex_for(dtype)
        ctype = dtypes.ctype(dtype)
        ctype2 = dtypes.ctype(dtype2)

        uniform_float = UniformFloat(0, 1).get_module(dtype)
        normal_bm = NormalBM(mean=0, std=1).get_module(dtype)

        self.module = Module(
            TEMPLATE.get_def("gamma"),
            render_kwds=dict(
                ctype=ctype, ctype2=ctype2,
                shape=shape, scale=dtypes.c_constant(scale, dtype),
                uniform_float=uniform_float, normal_bm=normal_bm))
