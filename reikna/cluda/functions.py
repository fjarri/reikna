"""
This module contains :py:class:`~reikna.cluda.Module` factories
which are used to compensate for the lack of complex number operations in OpenCL,
and the lack of C++ synthax which would allow one to write them.
"""

from warnings import warn

import numpy

from reikna.helpers import template_for
from reikna.cluda import dtypes
from reikna.cluda import Module


TEMPLATE = template_for(__file__)


def check_information_loss(out_dtype, expected_dtype):
    if dtypes.is_complex(expected_dtype) and not dtypes.is_complex(out_dtype):
        warn("Imaginary part ignored during the downcast from " +
            str(expected_dtype) + " to " + str(out_dtype),
            numpy.ComplexWarning)


def derive_out_dtype(out_dtype, *in_dtypes):
    expected_dtype = dtypes.result_type(*in_dtypes)
    if out_dtype is None:
        out_dtype = expected_dtype
    else:
        check_information_loss(out_dtype, expected_dtype)
    return out_dtype


def cast(out_dtype, in_dtype):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of one argument
    that casts values of ``in_dtype`` to ``out_dtype``.
    """
    return Module(
        TEMPLATE.get_def('cast'),
        render_kwds=dict(out_dtype=out_dtype, in_dtype=in_dtype))


def add(*in_dtypes, **kwds):
    """add(*in_dtypes, out_dtype=None)

    Returns a :py:class:`~reikna.cluda.Module`  with a function of
    ``len(in_dtypes)`` arguments that adds values of types ``in_dtypes``.
    If ``out_dtype`` is given, it will be set as a return type for this function.

    This is necessary since on some platforms the ``+`` operator for a complex and a real number
    works in an unexpected way (returning ``(a.x + b, a.y + b)`` instead of ``(a.x + b, a.y)``).
    """
    assert set(kwds.keys()).issubset(['out_dtype'])
    out_dtype = derive_out_dtype(kwds.get('out_dtype', None), *in_dtypes)
    return Module(
        TEMPLATE.get_def('add_or_mul'),
        render_kwds=dict(op='add', out_dtype=out_dtype, in_dtypes=in_dtypes))


def mul(*in_dtypes, **kwds):
    """mul(*in_dtypes, out_dtype=None)

    Returns a :py:class:`~reikna.cluda.Module`  with a function of
    ``len(in_dtypes)`` arguments that multiplies values of types ``in_dtypes``.
    If ``out_dtype`` is given, it will be set as a return type for this function.
    """
    assert set(kwds.keys()).issubset(['out_dtype'])
    out_dtype = derive_out_dtype(kwds.get('out_dtype', None), *in_dtypes)
    return Module(
        TEMPLATE.get_def('add_or_mul'),
        render_kwds=dict(op='mul', out_dtype=out_dtype, in_dtypes=in_dtypes))


def div(in_dtype1, in_dtype2, out_dtype=None):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of two arguments
    that divides values of ``in_dtype1`` and ``in_dtype2``.
    If ``out_dtype`` is given, it will be set as a return type for this function.
    """
    out_dtype = derive_out_dtype(out_dtype, in_dtype1, in_dtype2)
    return Module(
        TEMPLATE.get_def('div'),
        render_kwds=dict(out_dtype=out_dtype, in_dtype1=in_dtype1, in_dtype2=in_dtype2))


def conj(dtype):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of one argument
    that conjugates the value of type ``dtype`` (must be a complex data type).
    """
    if not dtypes.is_complex(dtype):
        raise NotImplementedError("conj() of " + str(dtype) + " is not supported")

    return Module(
        TEMPLATE.get_def('conj'),
        render_kwds=dict(dtype=dtype))


def polar_unit(dtype):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of one argument
    that returns a complex number ``(cos(theta), sin(theta))``
    for a value ``theta`` of type ``dtype`` (must be a real data type).
    """
    if not dtypes.is_real(dtype):
        raise NotImplementedError("polar_unit() of " + str(dtype) + " is not supported")

    return Module(
        TEMPLATE.get_def('polar_unit'),
        render_kwds=dict(dtype=dtype))


def norm(dtype):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of one argument
    that returns the 2-norm of the value of type ``dtype``
    (product by the complex conjugate if the value is complex, square otherwise).
    """
    return Module(
        TEMPLATE.get_def('norm'),
        render_kwds=dict(dtype=dtype))


def exp(dtype):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of one argument
    that exponentiates the value of type ``dtype``
    (must be a real or complex data type).
    """
    if dtypes.is_integer(dtype):
        raise NotImplementedError("exp() of " + str(dtype) + " is not supported")

    if dtypes.is_real(dtype):
        polar_unit_ = None
    else:
        polar_unit_ = polar_unit(dtypes.real_for(dtype))
    return Module(
        TEMPLATE.get_def('exp'),
        render_kwds=dict(dtype=dtype, polar_unit_=polar_unit_))


def pow(dtype, power_dtype=None):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of two arguments
    that raises the first argument of type ``dtype`` (must be a real or complex data type)
    to the power of the second argument (a corresponding real data type or an integer).
    """
    if dtypes.is_complex(power_dtype):
        raise NotImplementedError("pow() with a complex power is not supported")

    if power_dtype is None:
        if dtypes.is_integer(dtype):
            raise ValueError("Power dtype must be specified for an integer argument")
        elif dtypes.is_real(dtype):
            power_dtype = dtype
        else:
            power_dtype = dtypes.real_for(dtype)

    if dtypes.is_complex(dtype):
        r_dtype = dtypes.real_for(dtype)
    elif dtypes.is_real(dtype):
        r_dtype = dtype
    elif dtypes.is_real(power_dtype):
        r_dtype = power_dtype
    else:
        r_dtype = numpy.float32

    if dtypes.is_integer(dtype) and dtypes.is_real(power_dtype):
        dtype = power_dtype

    return Module(
        TEMPLATE.get_def('pow'),
        render_kwds=dict(
            dtype=dtype, power_dtype=power_dtype,
            mul_=mul(dtype, dtype), div_=div(dtype, dtype),
            polar_=polar(r_dtype)))


def polar(dtype):
    """
    Returns a :py:class:`~reikna.cluda.Module` with a function of two arguments
    that returns the complex-valued ``rho * exp(i * theta)``
    for values ``rho, theta`` of type ``dtype`` (must be a real data type).
    """
    if not dtypes.is_real(dtype):
        raise NotImplementedError("polar() of " + str(dtype) + " is not supported")

    return Module(
        TEMPLATE.get_def('polar'),
        render_kwds=dict(dtype=dtype, polar_unit_=polar_unit(dtype)))
