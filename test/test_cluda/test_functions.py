import itertools
from warnings import catch_warnings, filterwarnings

import pytest

import reikna.cluda as cluda
import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.helpers import product

from helpers import *


def get_func_kernel(thr, func_module, out_dtype, in_dtypes):
    src = """
    <%
        argnames = ["a" + str(i + 1) for i in range(len(in_dtypes))]
        in_ctypes = list(map(dtypes.ctype, in_dtypes))
        out_ctype = dtypes.ctype(out_dtype)
    %>
    KERNEL void test(
        GLOBAL_MEM ${out_ctype} *dest
        %for arg, ctype in zip(argnames, in_ctypes):
        , GLOBAL_MEM ${ctype} *${arg}
        %endfor
        )
    {
        const SIZE_T i = get_global_id(0);
        %for arg, ctype in zip(argnames, in_ctypes):
        ${ctype} ${arg}_load = ${arg}[i];
        %endfor

        dest[i] = ${func}(${", ".join([arg + "_load" for arg in argnames])});
    }
    """

    program = thr.compile(
        src,
        render_kwds=dict(in_dtypes=in_dtypes, out_dtype=out_dtype, func=func_module))

    return program.test


def generate_dtypes(out_code, in_codes):
    test_dtype = lambda idx: dict(i=numpy.int32, f=numpy.float32, c=numpy.complex64)[idx]
    in_dtypes = list(map(test_dtype, in_codes))
    out_dtype = dtypes.result_type(*in_dtypes) if out_code == 'auto' else test_dtype(out_code)

    if not any(map(dtypes.is_double, in_dtypes)):
        # numpy thinks that int32 * float32 == float64,
        # but we still need to run this test on older videocards
        if dtypes.is_complex(out_dtype):
            out_dtype = numpy.complex64
        elif dtypes.is_real(out_dtype):
            out_dtype = numpy.float32

    return out_dtype, in_dtypes


def check_func(thr, func_module, reference_func, out_dtype, in_dtypes):
    N = 256

    test = get_func_kernel(thr, func_module, out_dtype, in_dtypes)

    arrays = [get_test_array(N, dt, no_zeros=True, high=8) for dt in in_dtypes]
    arrays_dev = map(thr.to_device, arrays)
    dest_dev = thr.array(N, out_dtype)

    test(dest_dev, *arrays_dev, global_size=N)
    assert diff_is_negligible(
        thr.from_device(dest_dev),
        reference_func(*arrays).astype(out_dtype))


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('f', 'f'), ('c', 'c')])
def test_exp(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(thr, functions.exp(in_dtypes[0]), numpy.exp, out_dtype, in_dtypes)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'c'), ('f', 'f'), ('c', 'ci'), ('f', 'fi'), ('i', 'ii'), ('f', 'if')])
def test_pow(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    if len(in_dtypes) == 1:
        func = functions.pow(in_dtypes[0])
        if dtypes.is_real(in_dtypes[0]):
            in_dtypes.append(in_dtypes[0])
        else:
            in_dtypes.append(dtypes.real_for(in_dtypes[0]))
    else:
        func = functions.pow(in_dtypes[0], power_dtype=in_dtypes[1])
    check_func(thr, func, numpy.power, out_dtype, in_dtypes)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'f')])
def test_polar_unit(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        thr, functions.polar_unit(in_dtypes[0]),
        lambda theta: numpy.exp(1j * theta), out_dtype, in_dtypes)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'ff')])
def test_polar(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        thr, functions.polar(in_dtypes[0]),
        lambda rho, theta: rho * numpy.exp(1j * theta), out_dtype, in_dtypes)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('f', 'c'), ('f', 'f'), ('i', 'i')])
def test_norm(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        thr, functions.norm(in_dtypes[0]),
        lambda x: numpy.abs(x) ** 2, out_dtype, in_dtypes)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'c')])
def test_conj(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        thr, functions.conj(in_dtypes[0]),
        numpy.conj, out_dtype, in_dtypes)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('c', 'f'), ('f', 'f'), ('c', 'c')])
def test_cast(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        thr, functions.cast(out_dtype, in_dtypes[0]),
        dtypes.cast(out_dtype), out_dtype, in_dtypes)


@pytest.mark.parametrize(
    ('out_code', 'in_codes'),
    [('f', 'ff'), ('c', 'cc'), ('c', 'cf'), ('c', 'fc'), ('f', 'if')])
def test_div(thr, out_code, in_codes):
    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)
    check_func(
        thr, functions.div(*in_dtypes, out_dtype=out_dtype),
        lambda x, y: dtypes.cast(out_dtype)(x / y), out_dtype, in_dtypes)


@pytest.mark.parametrize('in_codes', ["ii", "ff", "cc", "cfi", "ifccfi"])
@pytest.mark.parametrize('out_code', ["auto", "i", "f", "c"])
def test_multiarg_mul(thr, out_code, in_codes):
    """
    Checks multi-argument mul() with a variety of data types.
    """

    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)

    def reference_mul(*args):
        res = product(args)
        if not dtypes.is_complex(out_dtype) and dtypes.is_complex(res.dtype):
            res = res.real
        return res.astype(out_dtype)

    # Temporarily catching imaginary part truncation warnings
    with catch_warnings():
        filterwarnings("ignore", "", numpy.ComplexWarning)
        mul = functions.mul(*in_dtypes, out_dtype=out_dtype)

    check_func(thr, mul, reference_mul, out_dtype, in_dtypes)


@pytest.mark.parametrize('in_codes', ["ii", "ff", "cc", "cfi", "ifccfi"])
@pytest.mark.parametrize('out_code', ["auto", "i", "f", "c"])
def test_multiarg_add(thr, out_code, in_codes):
    """
    Checks multi-argument add() with a variety of data types.
    """

    out_dtype, in_dtypes = generate_dtypes(out_code, in_codes)

    def reference_add(*args):
        res = sum(args)
        if not dtypes.is_complex(out_dtype) and dtypes.is_complex(res.dtype):
            res = res.real
        return res.astype(out_dtype)

    # Temporarily catching imaginary part truncation warnings
    with catch_warnings():
        filterwarnings("ignore", "", numpy.ComplexWarning)
        mul = functions.add(*in_dtypes, out_dtype=out_dtype)

    check_func(thr, mul, reference_add, out_dtype, in_dtypes)
