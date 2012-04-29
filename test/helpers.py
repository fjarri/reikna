import numpy
from tigger.cluda import dtypes

def getTestArray(shape, dtype):
    dtype = numpy.dtype(dtype)
    get_arr = lambda: numpy.random.normal(size=shape).astype(dtype)

    if dtypes.is_complex(dtype):
        return get_arr() + 1j * get_arr()
    else:
        return get_arr()

def diff(m, m_ref):
    return numpy.linalg.norm(m - m_ref) / numpy.linalg.norm(m_ref)
