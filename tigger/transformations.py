import tigger.cluda.dtypes as dtypes
from tigger.core import *
from tigger import Transformation


# Identity transformation: Output = Input
identity = Transformation(
    inputs=1, outputs=1,
    code="${o1.store}(${i1.load});")


# Parametrized scaling: Output = Input * Param
scale_param = Transformation(
    inputs=1, outputs=1, scalars=1,
    derive_o_from_is=lambda i1, s1: [i1],
    derive_is_from_o=lambda o1: ([o1], [o1]),
    derive_i_from_os=lambda o1, s1: [o1],
    derive_os_from_i=lambda i1: ([i1], [i1]),
    code="${o1.store}(${func.mul(i1.dtype, s1.dtype, out=o1.dtype)}(${i1.load}, ${s1}));")


def scale_const(multiplier):
    dtype = min_scalar_type(multiplier)
    return Transformation(
        inputs=1, outputs=1,
        code="${o1.store}(${func.mul(i1.dtype, numpy." + str(dtype) + ", out=o1.dtype)}(" +
            "${i1.load}, " + dtypes.c_constant(multiplier, dtype=dtype) + "));")


split_complex = Transformation(
    inputs=1, outputs=2,
    derive_o_from_is=lambda i1: [dtypes.real_for(i1), dtypes.real_for(i1)],
    derive_is_from_o=lambda o1, o2: ([dtypes.complex_for(o1)], []),
    derive_i_from_os=lambda o1, o2: [dtypes.complex_for(o1)],
    derive_os_from_i=lambda i1: ([dtypes.real_for(i1), dtypes.real_for(i1)], []),
    code="""
        ${o1.store}(${i1.load}.x);
        ${o2.store}(${i1.load}.y);
    """)


combine_complex = Transformation(
    inputs=2, outputs=1,
    derive_o_from_is=lambda i1, i2: [dtypes.complex_for(i1)],
    derive_is_from_o=lambda o1: ([dtypes.real_for(o1), dtypes.real_for(o1)], []),
    derive_i_from_os=lambda o1: [dtypes.real_for(o1), dtypes.real_for(o1)],
    derive_os_from_i=lambda i1, i2: ([dtypes.complex_for(o1)], []),
    code="${o1.store}(COMPLEX_CTR(${i1.load}, ${i2.load}));")
