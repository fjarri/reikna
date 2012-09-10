"""
This module contains a number of pre-created transformations.
"""

import tigger.cluda.dtypes as dtypes
from tigger.core import *
from tigger import Transformation


def identity():
    """
    Returns an identity transformation (1 output, 1 input): ``output1 = input1``.
    """
    return Transformation(
        inputs=1, outputs=1,
        code="${o1.store}(${i1.load});")


def scale_param():
    """
    Returns a scaling transformation with dynamic parameter (1 output, 1 input, 1 scalar):
    ``output1 = input1 * scalar1``.
    """
    return Transformation(
        inputs=1, outputs=1, scalars=1,
        derive_o_from_is=lambda i1, s1: [i1],
        derive_is_from_o=lambda o1: ([o1], [o1]),
        derive_i_from_os=lambda o1, s1: [o1],
        derive_os_from_i=lambda i1: ([i1], [i1]),
        code="${o1.store}(${func.mul(i1.dtype, s1.dtype, out=o1.dtype)}(${i1.load}, ${s1}));")


def scale_const(multiplier):
    """
    Returns a scaling transformation with fixed parameter (1 output, 1 input):
    ``output1 = input1 * <multiplier>``.
    """
    dtype = dtypes.min_scalar_type(multiplier)
    return Transformation(
        inputs=1, outputs=1,
        code="${o1.store}(${func.mul(i1.dtype, numpy." + str(dtype) + ", out=o1.dtype)}(" +
            "${i1.load}, " + dtypes.c_constant(multiplier, dtype=dtype) + "));")


def split_complex():
    """
    Returns a transformation which splits complex input into two real outputs
    (2 outputs, 1 input): ``output1 = Re(input1), output2 = Im(input1)``.
    """
    return Transformation(
        inputs=1, outputs=2,
        derive_o_from_is=lambda i1: [dtypes.real_for(i1), dtypes.real_for(i1)],
        derive_is_from_o=lambda o1, o2: ([dtypes.complex_for(o1)], []),
        derive_i_from_os=lambda o1, o2: [dtypes.complex_for(o1)],
        derive_os_from_i=lambda i1: ([dtypes.real_for(i1), dtypes.real_for(i1)], []),
        code="""
            ${o1.store}(${i1.load}.x);
            ${o2.store}(${i1.load}.y);
        """)


def combine_complex():
    """
    Returns a transformation which joins two real inputs into complex output
    (1 output, 2 inputs): ``output = input1 + 1j * input2``.
    """
    return Transformation(
        inputs=2, outputs=1,
        derive_o_from_is=lambda i1, i2: [dtypes.complex_for(i1)],
        derive_is_from_o=lambda o1: ([dtypes.real_for(o1), dtypes.real_for(o1)], []),
        derive_i_from_os=lambda o1: [dtypes.real_for(o1), dtypes.real_for(o1)],
        derive_os_from_i=lambda i1, i2: ([dtypes.complex_for(o1)], []),
        code="${o1.store}(COMPLEX_CTR(${o1.ctype})(${i1.load}, ${i2.load}));")
