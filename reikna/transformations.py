"""
This module contains a number of pre-created transformations.
"""

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.core import Transformation, Parameter, Annotation, Type


def copy(arr_t, out_arr_t=None):
    """
    Returns an identity transformation (1 output, 1 input): ``output = input``.
    Output array type ``out_arr_t`` may have different strides,
    but must have the same shape and data type.
    """
    if out_arr_t is None:
        out_arr_t = arr_t
    else:
        if out_arr_t.shape != arr_t.shape or out_arr_t.dtype != arr_t.dtype:
            raise ValueError("Input and output arrays must have the same shape and data type")

    return Transformation(
        [Parameter('output', Annotation(out_arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_same}(${input.load_same});")


def copy_broadcasted(arr_t, out_arr_t=None):
    """
    Returns an identity transformation (1 output, 1 input): ``output = input``,
    where ``input`` may be broadcasted (with the same semantics as ``numpy.broadcast_to()``).
    Output array type ``out_arr_t`` may have different strides,
    but must have compatible shapes the same shape and data type.

    .. note::

        This is an input-only transformation.
    """

    if out_arr_t is None:
        out_arr_t = arr_t

    if out_arr_t.dtype != arr_t.dtype:
        raise ValueError("Input and output arrays must have the same data type")

    in_tp = Type.from_value(arr_t)
    out_tp = Type.from_value(out_arr_t)
    if not in_tp.broadcastable_to(out_tp):
        raise ValueError("Input is not broadcastable to output")

    return Transformation(
        [Parameter('output', Annotation(out_arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        """
        ${output.store_same}(${input.load_idx}(
        %for i in range(len(input.shape)):
            %if input.shape[i] == 1:
            0
            %else:
            ${idxs[i + len(output.shape) - len(input.shape)]}
            %endif
            %if i != len(input.shape) - 1:
                ,
            %endif
        %endfor
        ));
        """,
        connectors=['output'])


def cast(arr_t, dtype):
    """
    Returns a typecast transformation of ``arr_t`` to ``dtype``
    (1 output, 1 input): ``output = cast[dtype](input)``.
    """
    dest = Type.from_value(arr_t).with_dtype(dtype)
    return Transformation(
        [Parameter('output', Annotation(dest, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_same}(${cast}(${input.load_same}));",
        render_kwds=dict(cast=functions.cast(dtype, arr_t.dtype)))


def add_param(arr_t, param_dtype):
    """
    Returns an addition transformation with a dynamic parameter (1 output, 1 input, 1 scalar):
    ``output = input + param``.
    """
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i')),
        Parameter('param', Annotation(param_dtype))],
        "${output.store_same}(${add}(${input.load_same}, ${param}));",
        render_kwds=dict(add=functions.add(arr_t.dtype, param_dtype, out_dtype=arr_t.dtype)))


def add_const(arr_t, param):
    """
    Returns an addition transformation with a fixed parameter (1 output, 1 input):
    ``output = input + param``.
    """
    param_dtype = dtypes.detect_type(param)
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_same}(${add}(${input.load_same}, ${param}));",
        render_kwds=dict(
            add=functions.add(arr_t.dtype, param_dtype, out_dtype=arr_t.dtype),
            param=dtypes.c_constant(param, dtype=param_dtype)))


def mul_param(arr_t, param_dtype):
    """
    Returns a scaling transformation with a dynamic parameter (1 output, 1 input, 1 scalar):
    ``output = input * param``.
    """
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i')),
        Parameter('param', Annotation(param_dtype))],
        "${output.store_same}(${mul}(${input.load_same}, ${param}));",
        render_kwds=dict(mul=functions.mul(arr_t.dtype, param_dtype, out_dtype=arr_t.dtype)))


def mul_const(arr_t, param):
    """
    Returns a scaling transformation with a fixed parameter (1 output, 1 input):
    ``output = input * param``.
    """
    param_dtype = dtypes.detect_type(param)
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_same}(${mul}(${input.load_same}, ${param}));",
        render_kwds=dict(
            mul=functions.mul(arr_t.dtype, param_dtype, out_dtype=arr_t.dtype),
            param=dtypes.c_constant(param, dtype=param_dtype)))


def div_param(arr_t, param_dtype):
    """
    Returns a scaling transformation with a dynamic parameter (1 output, 1 input, 1 scalar):
    ``output = input / param``.
    """
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i')),
        Parameter('param', Annotation(param_dtype))],
        "${output.store_same}(${div}(${input.load_same}, ${param}));",
        render_kwds=dict(div=functions.div(arr_t.dtype, param_dtype, out_dtype=arr_t.dtype)))


def div_const(arr_t, param):
    """
    Returns a scaling transformation with a fixed parameter (1 output, 1 input):
    ``output = input / param``.
    """
    param_dtype = dtypes.detect_type(param)
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
        "${output.store_same}(${div}(${input.load_same}, ${param}));",
        render_kwds=dict(
            div=functions.div(arr_t.dtype, param_dtype, out_dtype=arr_t.dtype),
            param=dtypes.c_constant(param, dtype=param_dtype)))


def split_complex(input_arr_t):
    """
    Returns a transformation that splits complex input into two real outputs
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
    Returns a transformation that joins two real inputs into complex output
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


def norm_const(arr_t, order):
    """
    Returns a transformation that calculates the ``order``-norm
    (1 output, 1 input): ``output = abs(input) ** order``.
    """
    if dtypes.is_complex(arr_t.dtype):
        out_dtype = dtypes.real_for(arr_t.dtype)
    else:
        out_dtype = arr_t.dtype

    return Transformation(
        [
            Parameter('output', Annotation(Type(out_dtype, arr_t.shape), 'o')),
            Parameter('input', Annotation(arr_t, 'i'))],
        """
        ${input.ctype} val = ${input.load_same};
        ${output.ctype} norm = ${norm}(val);
        %if order != 2:
        norm = pow(norm, ${dtypes.c_constant(order / 2, output.dtype)});
        %endif
        ${output.store_same}(norm);
        """,
        render_kwds=dict(
            norm=functions.norm(arr_t.dtype),
            order=order))


def norm_param(arr_t):
    """
    Returns a transformation that calculates the ``order``-norm
    (1 output, 1 input, 1 param): ``output = abs(input) ** order``.
    """
    if dtypes.is_complex(arr_t.dtype):
        out_dtype = dtypes.real_for(arr_t.dtype)
    else:
        out_dtype = arr_t.dtype

    return Transformation(
        [
            Parameter('output', Annotation(Type(out_dtype, arr_t.shape), 'o')),
            Parameter('input', Annotation(arr_t, 'i')),
            Parameter('order', Annotation(Type(out_dtype)))],
        """
        ${input.ctype} val = ${input.load_same};
        ${output.ctype} norm = ${norm}(val);
        norm = pow(norm, ${order} / 2);
        ${output.store_same}(norm);
        """,
        render_kwds=dict(
            norm=functions.norm(arr_t.dtype)))


def ignore(arr_t):
    """
    Returns a transformation that ignores the output it is attached to.
    """
    return Transformation(
        [Parameter('input', Annotation(arr_t, 'i'))],
        """
        // Ignoring intentionally
        """)


def broadcast_const(arr_t, val):
    """
    Returns a transformation that broadcasts the given constant to the array output
    (1 output): ``output = val``.
    """
    val = dtypes.cast(arr_t.dtype)(val)
    if len(val.shape) != 0:
        raise ValueError("The constant must be a scalar")
    return Transformation(
        [
            Parameter('output', Annotation(arr_t, 'o'))],
        """
        const ${output.ctype} val = ${dtypes.c_constant(val)};
        ${output.store_same}(val);
        """,
        render_kwds=dict(val=val))


def broadcast_param(arr_t):
    """
    Returns a transformation that broadcasts the free parameter to the array output
    (1 output, 1 param): ``output = param``.
    """
    return Transformation(
        [
            Parameter('output', Annotation(arr_t, 'o')),
            Parameter('param', Annotation(Type(arr_t.dtype)))],
        """
        ${output.store_same}(${param});
        """)
