import numpy


_DTYPE_TO_CTYPE = {}


def is_complex(dtype):
    """
    Returns ``True`` if ``dtype`` is complex.
    """
    dtype = normalize_type(dtype)
    return dtype.kind == 'c'

def is_double(dtype):
    """
    Returns ``True`` if ``dtype`` is double precision floating point.
    """
    dtype = normalize_type(dtype)
    return dtype.name in ['float64', 'complex128']

def is_integer(dtype):
    """
    Returns ``True`` if ``dtype`` is an integer.
    """
    dtype = normalize_type(dtype)
    return dtype.kind in ('i', 'u')

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
    """
    Wrapper for :py:func:`numpy.result_type`
    which takes into account types supported by GPUs.
    """
    return _promote_dtype(numpy.result_type(*dtypes))

def min_scalar_type(val):
    """
    Wrapper for :py:func:`numpy.min_scalar_dtype`
    which takes into account types supported by GPUs.
    """
    return _promote_dtype(numpy.min_scalar_type(val))

def normalize_type(dtype):
    """
    Function for wrapping all dtypes coming from the user.
    ``numpy`` uses two different classes to represent dtypes,
    and one of them does not have some important attributes.
    """
    return numpy.dtype(dtype)

def normalize_types(dtypes):
    """
    Same as :py:func:`normalize_type`, but operates on a list of dtypes.
    """
    return [normalize_type(dtype) for dtype in dtypes]

def ctype(dtype):
    """
    Returns C type name corresponding to given ``dtype``.
    """
    return _DTYPE_TO_CTYPE[normalize_type(dtype)]

def complex_for(dtype):
    """
    Returns complex dtype corresponding to given floating point ``dtype``.
    """
    dtype = normalize_type(dtype)
    return numpy.dtype(dict(float32='complex64', float64='complex128')[dtype.name])

def real_for(dtype):
    """
    Returns floating point dtype corresponding to given complex ``dtype``.
    """
    dtype = normalize_type(dtype)
    return numpy.dtype(dict(complex64='float32', complex128='float64')[dtype.name])

def complex_ctr(dtype):
    """
    Returns name of the constructor for the given ``dtype``.
    """
    return 'COMPLEX_CTR(' + ctype(dtype) + ')'

def zero_ctr(dtype):
    """
    Returns the string with constructed zero value for the given ``dtype``.
    """
    if is_complex(dtype):
        return complex_ctr(dtype) + '(0, 0)'
    else:
        return '0'

def cast(dtype):
    """
    Returns function that takes one argument and casts it to ``dtype``.
    """
    return numpy.cast[dtype]

def c_constant(val, dtype=None):
    """
    Returns a C-style numerical constant.
    """
    if dtype is None:
        dtype = min_scalar_type(val)
    if is_complex(dtype):
        return "COMPLEX_CTR(" + ctype(dtype) + ")(" + \
            c_constant(val.real) + ", " + c_constant(val.imag) + ")"
    elif is_integer(dtype):
        return str(val) + ("L" if dtype.itemsize > 4 else "")
    else:
        return str(val) + ("f" if dtype.itemsize <= 4 else "")

def _register_dtype(dtype, ctype):
    dtype = normalize_type(dtype)
    _DTYPE_TO_CTYPE[dtype] = ctype

# Taken from compyte.dtypes
def _fill_dtype_registry(respect_windows=True):

    import sys
    import platform

    _register_dtype(numpy.bool, "bool")
    _register_dtype(numpy.int8, "char")
    _register_dtype(numpy.uint8, "unsigned char")
    _register_dtype(numpy.int16, "short")
    _register_dtype(numpy.uint16, "unsigned short")
    _register_dtype(numpy.int32, "int")
    _register_dtype(numpy.uint32, "unsigned int")

    # recommended by Python docs
    is_64bits = sys.maxsize > 2 ** 32

    if is_64bits:
        if platform.system == 'Windows' and respect_windows:
            i64_name = "long long"
        else:
            i64_name = "long"

        _register_dtype(numpy.int64, i64_name)
        _register_dtype(numpy.uint64, "unsigned %s" % i64_name)

        # http://projects.scipy.org/numpy/ticket/2017
        _register_dtype(numpy.uintp, "unsigned %s" % i64_name)
    else:
        _register_dtype(numpy.uintp, "unsigned")

    _register_dtype(numpy.float32, "float")
    _register_dtype(numpy.float64, "double")
    _register_dtype(numpy.complex64, "float2")
    _register_dtype(numpy.complex128, "double2")

_fill_dtype_registry()
