import numpy

def is_complex(dtype):
    return dtype.kind == 'c'

def is_double(dtype):
    return dtype.name in ['float64', 'complex128']

def ctype(dtype):
    return dict(
        float32='float',
        float64='double',
        complex64='float2',
        complex128='double2'
    )[dtype.name]

def complex_for_dtype(dtype):
    x = cast(dtype)(1)
    return (x + 1j * x).dtype

def complex_ctr(dtype):
    return 'COMPLEX_CTR(' + ctype(dtype) + ')'

def zero_ctr(dtype):
    if is_complex(dtype):
        return complex_ctr(dtype) + '(0, 0)'
    else:
        return '0'

def cast(dtype):
    return lambda x: numpy.array([x]).astype(dtype)[0]
