"""
This module contains a number of pre-created transformations.
"""

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.core import *


def identity(arr_t):
    """
    Returns an identity transformation (1 output, 1 input): ``output = input``.
    """
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_same}(${input.load_same});")


def scale_param(arr_t, coeff_dtype):
    """
    Returns a scaling transformation with dynamic parameter (1 output, 1 input, 1 scalar):
    ``output = input * coeff``.
    """
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i')),
        Parameter('coeff', Annotation(coeff_dtype))],
        "${output.store_same}(${mul}(${input.load_same}, ${coeff}));",
        render_kwds=dict(mul=functions.mul(arr_t.dtype, coeff_dtype, out_dtype=arr_t.dtype)))


def scale_const(arr_t, coeff):
    """
    Returns a scaling transformation with fixed parameter (1 output, 1 input):
    ``output = input * <coeff>``.
    """
    coeff_dtype = dtypes.detect_type(coeff)
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_same}(${mul}(${input.load_same}, ${coeff}));",
        render_kwds=dict(
            mul=functions.mul(arr_t.dtype, coeff_dtype, out_dtype=arr_t.dtype),
            coeff=dtypes.c_constant(coeff, dtype=coeff_dtype)))


def split_complex(input_arr_t):
    """
    Returns a transformation which splits complex input into two real outputs
    (2 outputs, 1 input): ``real = Re(input), imag = Im(input)``.
    """
    output_t = Type(dtypes.real_for(input_arr_t.dtype), shape=input_arr_t.shape)
    return Transformation(
        [Parameter('real', Annotation(output_t, 'o')),
        Parameter('imag', Annotation(output_t, 'o')),
        Parameter('input', Annotation(input_arr_t, 'i'))],
        """
            ${real.store_same}(${input.load_same}.x);
            ${imag.store_same}(${input.load_same}.y);
        """)


def combine_complex(output_arr_t):
    """
    Returns a transformation which joins two real inputs into complex output
    (1 output, 2 inputs): ``output = real + 1j * imag``.
    """
    input_t = Type(dtypes.real_for(output_arr_t.dtype), shape=output_arr_t.shape)
    return Transformation(
        [Parameter('output', Annotation(output_arr_t, 'o')),
        Parameter('real', Annotation(input_t, 'i')),
        Parameter('imag', Annotation(input_t, 'i'))],
        """
        ${output.store_same}(
            COMPLEX_CTR(${output.ctype})(
                ${real.load_same},
                ${imag.load_same}));
        """)
