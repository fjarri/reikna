import numpy

def is_complex(dtype):
    return dtype.name in ['complex64', 'complex128']

def is_double(dtype):
    return dtype.name in ['float64', 'complex128']

def ctype(dtype):
    return dict(
        uint8='unsigned char',
        float32='float',
        float64='double',
    )[dtype.name]

def complex_ctr(dtype):
    return 'COMPLEX_CTR(' + ctype(dtype) + ')'

def zero_ctr(dtype):
    if is_complex(dtype):
        return complex_ctr(dtype) + '(0, 0)'
    else:
        return '0'

def cast(dtype):
    return lambda x: numpy.array([x]).astype(dtype)[0]
