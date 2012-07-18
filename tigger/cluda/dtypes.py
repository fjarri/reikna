import numpy

def is_complex(dtype):
    return numpy.dtype(dtype).kind == 'c'

def is_double(dtype):
    return numpy.dtype(dtype).name in ['float64', 'complex128']

def _promote_dtype(dtype):
    # not all numpy datatypes are supported by GPU, so we may need to promote
    dtype = numpy.dtype(dtype)
    if dtype.kind == 'i' and dtype.itemsize < 4:
        return numpy.int32
    elif dtype.kind == 'f' and dtype.itemsize < 4:
        return numpy.float32
    elif dtype.kind == 'c' and dtype.itemsize < 8:
        return numpy.complex64
    else:
        return dtype

def result_type(*dtypes):
    return _promote_dtype(numpy.result_type(*dtypes))

def min_scalar_type(val):
    return _promote_dtype(numpy.min_scalar_type(val))

def ctype(dtype):
    return dict(
        int32='int',
        float32='float',
        float64='double',
        complex64='float2',
        complex128='double2'
    )[numpy.dtype(dtype).name]

def complex_for(dtype):
    return numpy.dtype(dict(float32='complex64', float64='complex128')[numpy.dtype(dtype).name])

def real_for(dtype):
    return numpy.dtype(dict(complex64='float32', complex128='float64')[numpy.dtype(dtype).name])

def complex_ctr(dtype):
    return 'COMPLEX_CTR(' + ctype(dtype) + ')'

def zero_ctr(dtype):
    if is_complex(dtype):
        return complex_ctr(dtype) + '(0, 0)'
    else:
        return '0'

def cast(dtype):
    return lambda x: numpy.array([x]).astype(dtype)[0]
