"""A collection of pre-created transformations."""

from typing import Any

import numpy
from grunnur import ArrayMetadata, AsArrayMetadata, dtypes, functions

from .core import Annotation, Parameter, Transformation, Type


def copy(arr_t: AsArrayMetadata, out_arr_t: AsArrayMetadata | None = None) -> Transformation:
    """
    Returns an identity transformation (1 output, 1 input): ``output = input``.
    Output array type ``out_arr_t`` may have different strides,
    but must have the same shape and data type.
    """
    input_ = arr_t.as_array_metadata()
    output = out_arr_t.as_array_metadata() if out_arr_t is not None else input_

    if output.shape != input_.shape or output.dtype != input_.dtype:
        raise ValueError("Input and output arrays must have the same shape and data type")

    return Transformation(
        [
            Parameter("output", Annotation(output, "o")),
            Parameter("input", Annotation(input_, "i")),
        ],
        "${output.store_same}(${input.load_same});",
    )


def copy_broadcasted(
    arr_t: AsArrayMetadata, out_arr_t: AsArrayMetadata | None = None
) -> Transformation:
    """
    Returns an identity transformation (1 output, 1 input): ``output = input``,
    where ``input`` may be broadcasted (with the same semantics as ``numpy.broadcast_to()``).
    Output array type ``out_arr_t`` may have different strides,
    but must have compatible shapes the same shape and data type.

    .. note::

        This is an input-only transformation.
    """
    input_ = arr_t.as_array_metadata()
    output = out_arr_t.as_array_metadata() if out_arr_t is not None else input_

    if output.dtype != input_.dtype:
        raise ValueError("Input and output arrays must have the same data type")

    in_tp = Type.from_value(input_)
    out_tp = Type.from_value(output)
    if not in_tp.broadcastable_to(out_tp):
        raise ValueError("Input is not broadcastable to output")

    return Transformation(
        [
            Parameter("output", Annotation(output, "o")),
            Parameter("input", Annotation(input_, "i")),
        ],
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
        connectors=["output"],
    )


def cast(arr_t: AsArrayMetadata, dtype: numpy.dtype[Any]) -> Transformation:
    """
    Returns a typecast transformation of ``arr_t`` to ``dtype``
    (1 output, 1 input): ``output = cast[dtype](input)``.
    """
    input_ = arr_t.as_array_metadata()
    output = input_.with_(dtype=dtype)
    return Transformation(
        [Parameter("output", Annotation(output, "o")), Parameter("input", Annotation(input_, "i"))],
        "${output.store_same}(${cast}(${input.load_same}));",
        render_kwds=dict(cast=functions.cast(dtype, input_.dtype)),
    )


def add_param(arr_t: AsArrayMetadata, param_dtype: numpy.dtype[Any]) -> Transformation:
    """
    Returns an addition transformation with a dynamic parameter (1 output, 1 input, 1 scalar):
    ``output = input + param``.
    """
    input_ = arr_t.as_array_metadata()
    return Transformation(
        [
            Parameter("output", Annotation(input_, "o")),
            Parameter("input", Annotation(input_, "i")),
            Parameter("param", Annotation(param_dtype)),
        ],
        "${output.store_same}(${add}(${input.load_same}, ${param}));",
        render_kwds=dict(add=functions.add(input_.dtype, param_dtype, out_dtype=input_.dtype)),
    )


def add_const(arr_t: AsArrayMetadata, param: complex | numpy.number[Any]) -> Transformation:
    """
    Returns an addition transformation with a fixed parameter (1 output, 1 input):
    ``output = input + param``.
    """
    input_ = arr_t.as_array_metadata()
    param_dtype = dtypes.min_scalar_type(param)
    return Transformation(
        [Parameter("output", Annotation(input_, "o")), Parameter("input", Annotation(input_, "i"))],
        "${output.store_same}(${add}(${input.load_same}, ${param}));",
        render_kwds=dict(
            add=functions.add(input_.dtype, param_dtype, out_dtype=input_.dtype),
            param=dtypes.c_constant(param, dtype=param_dtype),
        ),
    )


def mul_param(arr_t: AsArrayMetadata, param_dtype: numpy.dtype[Any]) -> Transformation:
    """
    Returns a scaling transformation with a dynamic parameter (1 output, 1 input, 1 scalar):
    ``output = input * param``.
    """
    input_ = arr_t.as_array_metadata()
    return Transformation(
        [
            Parameter("output", Annotation(input_, "o")),
            Parameter("input", Annotation(input_, "i")),
            Parameter("param", Annotation(param_dtype)),
        ],
        "${output.store_same}(${mul}(${input.load_same}, ${param}));",
        render_kwds=dict(mul=functions.mul(input_.dtype, param_dtype, out_dtype=input_.dtype)),
    )


def mul_const(arr_t: AsArrayMetadata, param: complex | numpy.number[Any]) -> Transformation:
    """
    Returns a scaling transformation with a fixed parameter (1 output, 1 input):
    ``output = input * param``.
    """
    input_ = arr_t.as_array_metadata()
    param_dtype = dtypes.min_scalar_type(param)
    return Transformation(
        [Parameter("output", Annotation(input_, "o")), Parameter("input", Annotation(input_, "i"))],
        "${output.store_same}(${mul}(${input.load_same}, ${param}));",
        render_kwds=dict(
            mul=functions.mul(input_.dtype, param_dtype, out_dtype=input_.dtype),
            param=dtypes.c_constant(param, dtype=param_dtype),
        ),
    )


def div_param(arr_t: AsArrayMetadata, param_dtype: numpy.dtype[Any]) -> Transformation:
    """
    Returns a scaling transformation with a dynamic parameter (1 output, 1 input, 1 scalar):
    ``output = input / param``.
    """
    input_ = arr_t.as_array_metadata()
    return Transformation(
        [
            Parameter("output", Annotation(input_, "o")),
            Parameter("input", Annotation(input_, "i")),
            Parameter("param", Annotation(param_dtype)),
        ],
        "${output.store_same}(${div}(${input.load_same}, ${param}));",
        render_kwds=dict(div=functions.div(input_.dtype, param_dtype, out_dtype=input_.dtype)),
    )


def div_const(arr_t: AsArrayMetadata, param: complex | numpy.number[Any]) -> Transformation:
    """
    Returns a scaling transformation with a fixed parameter (1 output, 1 input):
    ``output = input / param``.
    """
    input_ = arr_t.as_array_metadata()
    param_dtype = dtypes.min_scalar_type(param)
    return Transformation(
        [Parameter("output", Annotation(input_, "o")), Parameter("input", Annotation(input_, "i"))],
        "${output.store_same}(${div}(${input.load_same}, ${param}));",
        render_kwds=dict(
            div=functions.div(input_.dtype, param_dtype, out_dtype=input_.dtype),
            param=dtypes.c_constant(param, dtype=param_dtype),
        ),
    )


def split_complex(input_arr_t: AsArrayMetadata) -> Transformation:
    """
    Returns a transformation that splits complex input into two real outputs
    (2 outputs, 1 input): ``real = Re(input), imag = Im(input)``.
    """
    input_ = input_arr_t.as_array_metadata()
    output = ArrayMetadata(shape=input_.shape, dtype=dtypes.real_for(input_.dtype))
    return Transformation(
        [
            Parameter("real", Annotation(output, "o")),
            Parameter("imag", Annotation(output, "o")),
            Parameter("input", Annotation(input_, "i")),
        ],
        """
            ${real.store_same}(${input.load_same}.x);
            ${imag.store_same}(${input.load_same}.y);
        """,
    )


def combine_complex(output_arr_t: AsArrayMetadata) -> Transformation:
    """
    Returns a transformation that joins two real inputs into complex output
    (1 output, 2 inputs): ``output = real + 1j * imag``.
    """
    output = output_arr_t.as_array_metadata()
    input_ = ArrayMetadata(shape=output.shape, dtype=dtypes.real_for(output.dtype))
    return Transformation(
        [
            Parameter("output", Annotation(output, "o")),
            Parameter("real", Annotation(input_, "i")),
            Parameter("imag", Annotation(input_, "i")),
        ],
        """
        ${output.store_same}(
            COMPLEX_CTR(${output.ctype})(
                ${real.load_same},
                ${imag.load_same}));
        """,
    )


def norm_const(arr_t: AsArrayMetadata, order: float) -> Transformation:
    """
    Returns a transformation that calculates the ``order``-norm
    (1 output, 1 input): ``output = abs(input) ** order``.
    """
    input_ = arr_t.as_array_metadata()
    out_dtype = dtypes.real_for(input_.dtype) if dtypes.is_complex(input_.dtype) else input_.dtype

    return Transformation(
        [
            Parameter(
                "output", Annotation(ArrayMetadata(shape=input_.shape, dtype=out_dtype), "o")
            ),
            Parameter("input", Annotation(input_, "i")),
        ],
        """
        ${input.ctype} val = ${input.load_same};
        ${output.ctype} norm = ${norm}(val);
        %if order != 2:
        norm = pow(norm, ${dtypes.c_constant(order / 2, output.dtype)});
        %endif
        ${output.store_same}(norm);
        """,
        render_kwds=dict(norm=functions.norm(input_.dtype), order=order, dtypes=dtypes),
    )


def norm_param(arr_t: AsArrayMetadata) -> Transformation:
    """
    Returns a transformation that calculates the ``order``-norm
    (1 output, 1 input, 1 param): ``output = abs(input) ** order``.
    """
    input_ = arr_t.as_array_metadata()
    out_dtype = dtypes.real_for(input_.dtype) if dtypes.is_complex(input_.dtype) else input_.dtype

    return Transformation(
        [
            Parameter(
                "output", Annotation(ArrayMetadata(shape=input_.shape, dtype=out_dtype), "o")
            ),
            Parameter("input", Annotation(input_, "i")),
            Parameter("order", Annotation(out_dtype)),
        ],
        """
        ${input.ctype} val = ${input.load_same};
        ${output.ctype} norm = ${norm}(val);
        norm = pow(norm, ${order} / 2);
        ${output.store_same}(norm);
        """,
        render_kwds=dict(norm=functions.norm(input_.dtype)),
    )


def ignore(arr_t: AsArrayMetadata) -> Transformation:
    """Returns a transformation that ignores the output it is attached to."""
    return Transformation(
        [Parameter("input", Annotation(arr_t.as_array_metadata(), "i"))],
        """
        // Ignoring intentionally
        """,
    )


def broadcast_const(arr_t: AsArrayMetadata, val: Any) -> Transformation:
    """
    Returns a transformation that broadcasts the given constant to the array output
    (1 output): ``output = val``.
    """
    input_ = arr_t.as_array_metadata()
    val = numpy.asarray(val, input_.dtype)
    if len(val.shape) != 0:
        raise ValueError("The constant must be a scalar")
    return Transformation(
        [Parameter("output", Annotation(input_, "o"))],
        """
        const ${output.ctype} val = ${dtypes.c_constant(val)};
        ${output.store_same}(val);
        """,
        render_kwds=dict(val=val, dtypes=dtypes),
    )


def broadcast_param(arr_t: AsArrayMetadata) -> Transformation:
    """
    Returns a transformation that broadcasts the free parameter to the array output
    (1 output, 1 param): ``output = param``.
    """
    input_ = arr_t.as_array_metadata()
    return Transformation(
        [
            Parameter("output", Annotation(input_, "o")),
            Parameter("param", Annotation(Type.scalar(input_.dtype))),
        ],
        """
        ${output.store_same}(${param});
        """,
    )
