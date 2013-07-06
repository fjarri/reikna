"""
This module contains a number of pre-created transformations.
"""

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.core import *


def identity():
    """
    Returns an identity transformation (1 output, 1 input): ``output1 = input1``.
    """
    return Transformation(
        inputs=1, outputs=1,
        snippet="${o1.store}(${i1.load});")


def scale_param():
    """
    Returns a scaling transformation with dynamic parameter (1 output, 1 input, 1 scalar):
    ``output1 = input1 * scalar1``.
    """
    return Transformation(
        inputs=1, outputs=1, scalars=1,
        snippet="${o1.store}(${mul}(${i1.load}, ${s1}));",
        derive_render_kwds=lambda o1, i1, s1: dict(
            mul=functions.mul(i1, s1, out_dtype=o1)))


def scale_const(multiplier):
    """
    Returns a scaling transformation with fixed parameter (1 output, 1 input):
    ``output1 = input1 * <multiplier>``.
    """
    dtype = dtypes.detect_type(multiplier)
    return Transformation(
        inputs=1, outputs=1,
        snippet="${o1.store}(${mul}(${i1.load}, ${coeff}));",
        derive_render_kwds=lambda o1, i1: dict(
            mul=functions.mul(i1, dtype, out_dtype=o1),
            coeff=dtypes.c_constant(multiplier, dtype=dtype)))


def split_complex():
    """
    Returns a transformation which splits complex input into two real outputs
    (2 outputs, 1 input): ``out_re = Re(in), out_im = Im(in)``.
    """
    return Transformation(
        inputs=['in_c'], outputs=['out_re', 'out_im'],
        derive_i_from_os=lambda out_re, out_im: dtypes.complex_for(out_re),
        snippet="""
            ${out_re.store}(${in_c.load}.x);
            ${out_im.store}(${in_c.load}.y);
        """)


def combine_complex():
    """
    Returns a transformation which joins two real inputs into complex output
    (1 output, 2 inputs): ``out = in_re + 1j * in_im``.
    """
    return Transformation(
        inputs=['in_re', 'in_im'], outputs=['out_c'],
        derive_o_from_is=lambda in_re, in_im: dtypes.complex_for(in_re),
        snippet="${out_c.store}(COMPLEX_CTR(${out_c.ctype})(${in_re.load}, ${in_im.load}));")
