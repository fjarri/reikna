from __future__ import annotations

from typing import Any, Callable

import numpy
from grunnur import Module, Template, dtypes, functions

import reikna.helpers as helpers

from .bijections import Bijection

TEMPLATE = Template.from_associated_file(__file__)


class Sampler:
    """
    Contains a random distribution sampler module and accompanying metadata.
    Supports ``__process_modules__`` protocol.

    .. py:attribute:: deterministic

        If ``True``, every sampled random number consumes the same amount of counters.

    .. py:attribute:: randoms_per_call

        How many random numbers one call to ``sample`` creates.

    .. py:attribute:: dtype

        The data type of one random value produced by the sampler.

    .. py:attribute:: module

        The module containing the distribution sampling function.
        It provides the C functions below.

    .. c:macro:: RANDOMS_PER_CALL

        Contains the value of :py:attr:`randoms_per_call`.

    .. c:type:: Value

        Contains the type corresponding to :py:attr:`dtype`.

    .. c:type:: Result

        Describes the sampling result.

        .. c:member:: value v[RANDOMS_PER_CALL]

    .. c:type:: Result sample(State *state)

        Performs the sampling, updating the state.
    """

    def __init__(
        self,
        bijection: Bijection,
        module: Module,
        dtype: numpy.dtype[Any],
        randoms_per_call: int = 1,
        deterministic: bool = False,
    ):
        """__init__()"""  # hide the signature from Sphinx
        self.randoms_per_call = randoms_per_call
        self.dtype = numpy.dtype(dtype)
        self.deterministic = deterministic
        self.bijection = bijection
        self.module = module

    def __process_modules__(self, process: Callable[[Any], Any]) -> Sampler:
        return Sampler(
            process(self.bijection),
            process(self.module),
            self.dtype,
            randoms_per_call=self.randoms_per_call,
            deterministic=self.deterministic,
        )


def uniform_integer(
    bijection: Bijection, dtype: numpy.dtype[Any], low: int, high: int | None = None
) -> Sampler:
    """
    Generates uniformly distributed integer numbers in the interval ``[low, high)``.
    If ``high`` is ``None``, the interval is ``[0, low)``.
    Supported dtypes: any numpy integers.
    If the size of the interval is a power of 2, a fixed number of counters
    is used in each thread.
    Returns a :py:class:`~reikna.cbrng.samplers.Sampler` object.
    """

    if high is None:
        low, high = 0, low + 1
    else:
        assert low < high - 1

    dtype = numpy.dtype(dtype)
    ctype = dtypes.ctype(dtype)

    if dtype.kind == "i":
        assert low >= -(2 ** (dtype.itemsize * 8 - 1))
        assert high < 2 ** (dtype.itemsize * 8 - 1)
    else:
        assert low >= 0
        assert high < 2 ** (dtype.itemsize * 8)

    num = high - low
    raw_dtype: numpy.dtype[numpy.unsignedinteger[Any]]
    if num <= 2**32:
        raw_dtype = numpy.dtype("uint32")
    else:
        raw_dtype = numpy.dtype("uint64")

    raw_func = bijection.raw_functions[raw_dtype]
    max_num = 2 ** (raw_dtype.itemsize * 8)

    raw_ctype = dtypes.ctype(raw_dtype)

    module = Module(
        TEMPLATE.get_def("uniform_integer"),
        render_globals=dict(
            bijection=bijection,
            dtype=dtype,
            ctype=ctype,
            raw_ctype=raw_ctype,
            raw_func=raw_func,
            max_num=max_num,
            num=num,
            low=low,
        ),
    )

    return Sampler(bijection, module, dtype, deterministic=(max_num % num == 0))


def uniform_float(
    bijection: Bijection, dtype: numpy.dtype[Any], low: float = 0, high: float = 1
) -> Sampler:
    """
    Generates uniformly distributed floating-points numbers in the interval ``[low, high)``.
    Supported dtypes: ``float(32/64)``.
    A fixed number of counters is used in each thread.
    Returns a :py:class:`~reikna.cbrng.samplers.Sampler` object.
    """
    assert low < high

    ctype = dtypes.ctype(dtype)

    bitness = 64 if dtypes.is_double(dtype) else 32
    raw_func = "get_raw_uint" + str(bitness)
    raw_max = dtypes.c_constant(2**bitness, dtype)

    size_const = dtypes.c_constant(high - low, dtype)
    low_const = dtypes.c_constant(low, dtype)

    module = Module(
        TEMPLATE.get_def("uniform_float"),
        render_globals=dict(
            bijection=bijection,
            ctype=ctype,
            raw_func=raw_func,
            raw_max=raw_max,
            size=size_const,
            low=low_const,
        ),
    )

    return Sampler(bijection, module, dtype, deterministic=True)


def normal_bm(
    bijection: Bijection, dtype: numpy.dtype[Any], mean: float = 0, std: float = 1
) -> Sampler:
    """
    Generates normally distributed random numbers with the mean ``mean`` and
    the standard deviation ``std`` using Box-Muller transform.
    Supported dtypes: ``float(32/64)``, ``complex(64/128)``.
    Produces two random numbers per call for real types and one number for complex types.
    Returns a :py:class:`~reikna.cbrng.samplers.Sampler` object.

    .. note::

        In case of a complex ``dtype``, ``std`` refers to the standard deviation of the
        complex numbers (same as ``numpy.std()`` returns), not real and imaginary components
        (which will be normally distributed with the standard deviation ``std / sqrt(2)``).
        Consequently, while ``mean`` is of type ``dtype``, ``std`` must be real.
    """

    if dtypes.is_complex(dtype):
        r_dtype = dtypes.real_for(dtype)
        c_dtype = dtype
    else:
        r_dtype = dtype
        c_dtype = dtypes.complex_for(dtype)

    uf = uniform_float(bijection, r_dtype, low=0, high=1)

    module = Module(
        TEMPLATE.get_def("normal_bm"),
        render_globals=dict(
            dtypes=dtypes,
            complex_res=dtypes.is_complex(dtype),
            r_dtype=r_dtype,
            r_ctype=dtypes.ctype(r_dtype),
            c_dtype=c_dtype,
            c_ctype=dtypes.ctype(c_dtype),
            polar_unit=functions.polar_unit(r_dtype),
            bijection=bijection,
            mean=mean,
            std=std,
            uf=uf,
        ),
    )

    return Sampler(
        bijection,
        module,
        dtype,
        deterministic=uf.deterministic,
        randoms_per_call=1 if dtypes.is_complex(dtype) else 2,
    )


def gamma(
    bijection: Bijection, dtype: numpy.dtype[Any], shape: float = 1, scale: float = 1
) -> Sampler:
    """
    Generates random numbers from the gamma distribution

    .. math::
      P(x) = x^{k-1} \\frac{e^{-x/\\theta}}{\\theta^k \\Gamma(k)},

    where :math:`k` is ``shape``, and :math:`\\theta` is ``scale``.
    Supported dtypes: ``float(32/64)``.
    Returns a :py:class:`~reikna.cbrng.samplers.Sampler` object.
    """

    ctype = dtypes.ctype(dtype)
    uf = uniform_float(bijection, dtype, low=0, high=1)
    nbm = normal_bm(bijection, dtype, mean=0, std=1)

    module = Module(
        TEMPLATE.get_def("gamma"),
        render_globals=dict(
            dtypes=dtypes,
            dtype=dtype,
            ctype=ctype,
            bijection=bijection,
            shape=shape,
            scale=dtypes.c_constant(scale, dtype),
            uf=uf,
            nbm=nbm,
        ),
    )

    return Sampler(bijection, module, dtype)


def vonmises(
    bijection: Bijection, dtype: numpy.dtype[Any], mu: float = 0, kappa: float = 1
) -> Sampler:
    """
    Generates random numbers from the von Mises distribution

    .. math::
      P(x) = \\frac{\\exp(\\kappa \\cos(x - \\mu))}{2 \\pi I_0(\\kappa)},

    where :math:`\\mu` is the mode, :math:`\\kappa` is the dispersion,
    and :math:`I_0` is the modified Bessel function of the first kind.
    Supported dtypes: ``float(32/64)``.
    Returns a :py:class:`~reikna.cbrng.samplers.Sampler` object.
    """

    ctype = dtypes.ctype(dtype)
    uf = uniform_float(bijection, dtype, low=0, high=1)

    module = Module(
        TEMPLATE.get_def("vonmises"),
        render_globals=dict(
            dtypes=dtypes,
            dtype=dtype,
            ctype=ctype,
            bijection=bijection,
            mu=dtypes.c_constant(mu, dtype),
            kappa=kappa,
            uf=uf,
        ),
    )

    return Sampler(bijection, module, dtype)


# List of samplers that can be used as convenience constructors in CBRNG class
SAMPLERS = {
    "uniform_integer": uniform_integer,
    "uniform_float": uniform_float,
    "normal_bm": normal_bm,
    "gamma": gamma,
    "vonmises": vonmises,
}
