import numpy
import pytest

from helpers import *
from reikna.pureparallel import PureParallel
from reikna.core import Parameter, Annotation, Type
import reikna.cluda.dtypes as dtypes
from reikna.transformations import scale_param


def test_guiding_input(thr):

    N = 1000
    dtype = numpy.float32

    p = PureParallel(
        [
            Parameter('output', Annotation(Type(dtype, shape=(2, N)), 'o')),
            Parameter('input', Annotation(Type(dtype, shape=N), 'i'))],
        """
        float t = ${input.load_idx}(${idxs[0]});
        ${output.store_idx}(0, ${idxs[0]}, t);
        ${output.store_idx}(1, ${idxs[0]}, t * 2);
        """,
        guiding_array='input')

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev)

    res_ref = numpy.vstack([a, a * 2])

    assert diff_is_negligible(res_dev.get(), res_ref)


def test_guiding_output(thr):

    N = 1000
    dtype = numpy.float32

    p = PureParallel(
        [
            Parameter('output', Annotation(Type(dtype, shape=N), 'o')),
            Parameter('input', Annotation(Type(dtype, shape=(2, N)), 'i'))],
        """
        float t1 = ${input.load_idx}(0, ${idxs[0]});
        float t2 = ${input.load_idx}(1, ${idxs[0]});
        ${output.store_idx}(${idxs[0]}, t1 + t2);
        """,
        guiding_array='output')

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev)

    res_ref = a[0] + a[1]

    assert diff_is_negligible(res_dev.get(), res_ref)


def test_guiding_shape(thr):

    N = 1000
    dtype = numpy.float32

    p = PureParallel(
        [
            Parameter('output', Annotation(Type(dtype, shape=(2, N)), 'o')),
            Parameter('input', Annotation(Type(dtype, shape=(2, N)), 'i'))],
        """
        float t1 = ${input.load_idx}(0, ${idxs[0]});
        float t2 = ${input.load_idx}(1, ${idxs[0]});
        ${output.store_idx}(0, ${idxs[0]}, t1 + t2);
        ${output.store_idx}(1, ${idxs[0]}, t1 - t2);
        """,
        guiding_array=(N,))

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev)

    res_ref = numpy.vstack([a[0] + a[1], a[0] - a[1]])

    assert diff_is_negligible(res_dev.get(), res_ref)


def test_trf_with_guiding_input(thr):
    """
    Test the creation of ``PureParallel`` out of a transformation,
    with an input parameter as a guiding array.
    """

    N = 1000
    coeff = 3
    dtype = numpy.float32

    arr_t = Type(dtype, shape=N)
    trf = scale_param(arr_t, dtype)
    p = PureParallel.from_trf(trf, trf.input)

    # The new PureParallel has to preserve the parameter list of the original transformation.
    assert list(p.signature.parameters.values()) == list(trf.signature.parameters.values())

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev, coeff)

    assert diff_is_negligible(res_dev.get(), a * 3)


def test_trf_with_guiding_output(thr):
    """
    Test the creation of ``PureParallel`` out of a transformation,
    with an output parameter as a guiding array.
    """

    N = 1000
    coeff = 3
    dtype = numpy.float32

    arr_t = Type(dtype, shape=N)
    trf = scale_param(arr_t, dtype)
    p = PureParallel.from_trf(trf, trf.output)

    # The new PureParallel has to preserve the parameter list of the original transformation.
    assert list(p.signature.parameters.values()) == list(trf.signature.parameters.values())

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev, coeff)

    assert diff_is_negligible(res_dev.get(), a * 3)
