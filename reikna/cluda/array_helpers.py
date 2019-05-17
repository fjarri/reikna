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
from reikna.core import Type, Parameter, Annotation, Computation


def normalize_value(thr, gpu_array_type, val):
    """
    Transforms a given value (a scalar or an array)
    to a value that can be passed to a kernel.
    Returns a pair (result, is_array), where ``is_array`` is a boolean
    that helps distinguishing between zero-dimensional arrays and scalars.
    """
    if isinstance(val, gpu_array_type):
        return val, True
    elif isinstance(val, numpy.ndarray):
        return thr.to_device(val), True
    else:
        dtype = dtypes.detect_type(val)
        return numpy.cast[dtype](val), False


def setitem_computation(dest, source, is_array):
    """
    Returns a compiled computation that broadcasts ``source`` to ``dest``,
    where ``dest`` is a GPU array, and ``source`` is either a GPU array or a scalar.
    """
    if is_array:
        source_dt = Type.from_value(source).with_dtype(dest.dtype)
        trf = transformations.copy(source_dt, dest)
        comp = PureParallel.from_trf(trf, guiding_array=trf.output)
        cast_trf = transformations.cast(source, dest.dtype)
        comp.parameter.input.connect(cast_trf, cast_trf.output, src_input=cast_trf.input)
        return comp
    else:
        trf = transformations.broadcast_param(dest)
        return PureParallel.from_trf(trf, guiding_array=trf.output)


def setitem_method(array, index, value):
    # We need it both in ``cuda.Array`` and ``ocl.Array``, hence a standalone function.

    # PyOpenCL and PyCUDA support __setitem__() for some restricted cases,
    # but it is too complicated to determine when it will work,
    # and it is easier to just call our own implementation every time.

    view = array[index]
    value, is_array = normalize_value(array.thread, type(array), value)
    comp = array.thread.get_cached_computation(
        setitem_computation, Type.from_value(view), Type.from_value(value), is_array)
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


def roll_computation(array, axis):
    return PureParallel(
        [
            Parameter('output', Annotation(array, 'o')),
            Parameter('input', Annotation(array, 'i')),
            Parameter('shift', Annotation(Type(numpy.int32)))],
        """
        <%
            shape = input.shape
        %>
        %for i in range(len(shape)):
            VSIZE_T output_${idxs[i]} =
                %if i == axis:
                ${shift} == 0 ?
                    ${idxs[i]} :
                    ## Since ``shift`` can be negative, and its absolute value greater than
                    ## ``shape[i]``, a double modulo division is necessary
                    ## (the ``%`` operator preserves the sign of the dividend in C).
                    (${idxs[i]} + (${shape[i]} + ${shift} % ${shape[i]})) % ${shape[i]};
                %else:
                ${idxs[i]};
                %endif
        %endfor
        ${output.store_idx}(
            ${", ".join("output_" + name for name in idxs)},
            ${input.load_idx}(${", ".join(idxs)}));
        """,
        guiding_array='input',
        render_kwds=dict(axis=axis))


class RollInplace(Computation):

    def __init__(self, array, axis):
        self._axis = axis
        Computation.__init__(self, [
            Parameter('array', Annotation(array, 'io')),
            Parameter('shift', Annotation(Type(numpy.int32)))])

    def _build_plan(self, plan_factory, device_params, array, shift):
        plan = plan_factory()

        temp = plan.temp_array_like(array)
        plan.computation_call(roll_computation(array, self._axis), temp, array, shift)

        tr = transformations.copy(temp, out_arr_t=array)
        copy_comp = PureParallel.from_trf(tr, guiding_array=tr.output)
        plan.computation_call(copy_comp, array, temp)

        return plan


def roll(array, shift, axis=-1):
    """
    Cyclically shifts elements of ``array`` by ``shift`` positions to the right along ``axis``.
    ``shift`` can be negative (in which case the elements are shifted to the left).
    Elements that are shifted beyond the last position are re-introduced at the first
    (and vice versa).

    Works equivalently to ``numpy.roll`` (except ``axis=None`` is not supported).
    """
    temp = array.thread.array(array.shape, array.dtype)
    axis = axis % len(array.shape)
    comp = array.thread.get_cached_computation(
        roll_computation, Type.from_value(array), axis)
    comp(temp, array, shift)
    return temp


def roll_method(array, shift, axis=-1):
    axis = axis % len(array.shape)
    comp = array.thread.get_cached_computation(
        RollInplace, Type.from_value(array), axis)
    comp(array, shift)
