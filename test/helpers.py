import numpy
from reikna.cluda import dtypes


SINGLE_EPS = 1e-6
DOUBLE_EPS = 1e-11


def get_test_array(shape, dtype, no_zeros=False, high=None):
    if not isinstance(shape, tuple):
        shape = (shape,)

    dtype = dtypes.normalize_type(dtype)

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
        return get_arr() + 1j * get_arr()
    else:
        return get_arr()

def float_diff(m, m_ref):
    return numpy.linalg.norm(m - m_ref) / numpy.linalg.norm(m_ref)

def diff_is_negligible(m, m_ref):
    assert m.dtype == m_ref.dtype

    if dtypes.is_integer(m.dtype):
        return ((m - m_ref) == 0).all()

    diff = float_diff(m, m_ref)
    if dtypes.is_double(m.dtype):
        return diff < DOUBLE_EPS
    else:
        return diff < SINGLE_EPS
