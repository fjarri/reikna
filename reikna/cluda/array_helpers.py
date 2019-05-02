# FIXME: This module is a part of Array functionality, so it is located on CLUDA level,
# but it requires some high-level Reikna functionality (computations and transformations).
# So it is a bit of circular dependency.
# Ideally, this should be moved to computation level, perhaps creating two versions of Array -
# CLUDA level (without __setitem__) and Reikna level (with one).

import numpy

import reikna.cluda.dtypes as dtypes
import reikna.transformations as transformations
import reikna.cluda.functions as functions
from reikna.algorithms import PureParallel
from reikna.core import Type


def normalize_value(thr, gpu_array_type, val):
    """
    Transforms a given value (a scalar or an array)
    to a value that can be passed to a kernel.
    """
    if isinstance(val, gpu_array_type):
        return val
    elif isinstance(val, numpy.ndarray):
        return thr.to_device(val)
    else:
        dtype = dtypes.detect_type(val)
        return numpy.cast[dtype](val)


def setitem_computation(dest, source):
    """
    Returns a compiled computation that broadcasts ``source`` to ``dest``,
    where ``dest`` is a GPU array, and ``source`` is either a GPU array or a scalar.
    """
    if len(source.shape) == 0:
        trf = transformations.broadcast_param(dest)
        return PureParallel.from_trf(trf, guiding_array=trf.output)
    else:
        source_dt = Type.from_value(source).with_dtype(dest.dtype)
        trf = transformations.copy(source_dt, dest)
        comp = PureParallel.from_trf(trf, guiding_array=trf.output)
        cast_trf = transformations.cast(source, dest.dtype)
        comp.parameter.input.connect(cast_trf, cast_trf.output, src_input=cast_trf.input)
        return comp


def setitem_method(array, index, value):
    # We need it both in ``cuda.Array`` and ``ocl.Array``, hence a standalone function.

    # PyOpenCL and PyCUDA support __setitem__() for some restricted cases,
    # but it is too complicated to determine when it will work,
    # and it is easier to just call our own implementation every time.

    view = array[index]
    value = normalize_value(array.thread, type(array), value)
    comp = array.thread.get_cached_computation(
        setitem_computation, Type.from_value(view), Type.from_value(value))
    comp(view, value)


def get_method(array):
    temp = array.thread.array(array.shape, array.dtype)
    comp = array.thread.get_cached_computation(
        setitem_computation, Type.from_value(temp), Type.from_value(array))
    comp(temp, array)
    return temp.get()


def is_shape_compatible(template_shape, shape, axis):
    for i in range(len(template_shape)):
        if i != axis and shape[i] != template_shape[i]:
            return False
    return True


def concatenate(arrays, axis=0, out=None):
    """
    Concatenate an iterable of arrays along ``axis`` and write them to ``out``
    (allocating it if it is set to ``None``).

    Works analogously to ``numpy.concatenate()`` (except ``axis=None`` is not supported).
    """

    # TODO: support axis=None.
    # Requires Array.ravel() returnign an Array instead of CUDA/PyOpenCL array.

    if len(arrays) == 0:
        raise ValueError("Need at least one array to concatenate")
    if any(array.dtype != arrays[0].dtype for array in arrays[1:]):
        raise ValueError("Data types of all arrays must be the same")

    dtype = arrays[0].dtype
    thread = arrays[0].thread

    template_shape = arrays[0].shape
    axis = axis % len(template_shape)
    for array in arrays[1:]:
        if not is_shape_compatible(template_shape, array.shape, axis):
            raise ValueError(
                "Shapes are not compatible: " + str(template_shape) + " and " + str(shape))

    out_shape = list(template_shape)
    out_shape[axis] = sum(array.shape[axis] for array in arrays)
    out_shape = tuple(out_shape)

    if out is None:
        out = thread.array(out_shape, dtype)
    else:
        if out.shape != out_shape:
            raise ValueError(
                "Incorrect output shape: expected " + str(out_shape) + ", got " + str(out.shape))
        if out.dtype != dtype:
            raise ValueError(
                "Incorrect output dtype: expected " + str(dtype) + ", got " + str(out.dtype))

    offset = 0
    slices = [slice(None) for i in range(len(out_shape))]
    for array in arrays:
        slices[axis] = slice(offset, offset + array.shape[axis])
        out[tuple(slices)] = array
        offset += array.shape[axis]

    return out
