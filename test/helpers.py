import numpy
from numpy.lib.stride_tricks import as_strided

from reikna.helpers import wrap_in_tuple
from reikna.cluda import dtypes

SINGLE_EPS = 1e-6
DOUBLE_EPS = 1e-11


def get_test_array_like(arr, **kwds):
    kwds['strides'] = arr.strides
    return get_test_array(arr.shape, arr.dtype, **kwds)

def get_test_array(shape, dtype, strides=None, no_zeros=False, high=None):
    shape = wrap_in_tuple(shape)
    dtype = dtypes.normalize_type(dtype)

    if dtype.names is not None:
        result = numpy.empty(shape, dtype)
        for name in dtype.names:
            result[name] = get_test_array(shape, dtype[name], no_zeros=no_zeros, high=high)
    else:
        if dtypes.is_integer(dtype):
            low = 1 if no_zeros else 0
            if high is None:
                high = 100 # will work even with signed chars
            get_arr = lambda: numpy.random.randint(low, high, shape).astype(dtype)
        else:
            low = 0.01 if no_zeros else 0
            if high is None:
                high = 1.0
            get_arr = lambda: numpy.random.uniform(low, high, shape).astype(dtype)

        if dtypes.is_complex(dtype):
            result = get_arr() + 1j * get_arr()
        else:
            result = get_arr()

    if strides is not None:
        result = as_strided(result, result.shape, strides)

    return result

def float_diff(m, m_ref):
    return numpy.linalg.norm(m - m_ref) / numpy.linalg.norm(m_ref)

def diff_is_negligible(m, m_ref):

    if m.dtype.names is not None:
        return all(diff_is_negligible(m[name], m_ref[name]) for name in m.dtype.names)

    assert m.dtype == m_ref.dtype

    if dtypes.is_integer(m.dtype):
        return ((m - m_ref) == 0).all()

    diff = float_diff(m, m_ref)
    if dtypes.is_double(m.dtype):
        return diff < DOUBLE_EPS
    else:
        return diff < SINGLE_EPS
