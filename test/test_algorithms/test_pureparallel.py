import numpy
import pytest

from helpers import *
from reikna.algorithms import PureParallel
from reikna.core import Parameter, Annotation, Type, Computation
import reikna.cluda.dtypes as dtypes
from reikna.transformations import mul_param, copy


class NestedPureParallel(Computation):

    def __init__(self, size, dtype):

        Computation.__init__(self, [
            Parameter('output', Annotation(Type(dtype, shape=size), 'o')),
            Parameter('input', Annotation(Type(dtype, shape=size), 'i'))])

        self._p = PureParallel([
                Parameter('output', Annotation(Type(dtype, shape=size), 'o')),
                Parameter('i1', Annotation(Type(dtype, shape=size), 'i')),
                Parameter('i2', Annotation(Type(dtype, shape=size), 'i'))],
            """
            ${i1.ctype} t1 = ${i1.load_idx}(${idxs[0]});
            ${i2.ctype} t2 = ${i2.load_idx}(${idxs[0]});
            ${output.store_idx}(${idxs[0]}, t1 + t2);
            """)

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()
        plan.computation_call(self._p, output, input_, input_)
        return plan


def test_nested(thr):

    N = 1000
    dtype = numpy.float32

    p = NestedPureParallel(N, dtype)

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev)

    res_ref = a + a

    assert diff_is_negligible(res_dev.get(), res_ref)


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


def test_zero_length_shape(thr):

    dtype = numpy.float32

    p = PureParallel(
        [
            Parameter('output', Annotation(Type(dtype, shape=tuple()), 'o')),
            Parameter('input', Annotation(Type(dtype, shape=tuple()), 'i'))],
        """
        float t = ${input.load_idx}();
        ${output.store_idx}(t * 2);
        """,
        guiding_array=tuple())

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev)

    res_ref = (a * 2).astype(dtype)

    assert diff_is_negligible(res_dev.get(), res_ref)


@pytest.mark.parametrize('guiding_array', ['input', 'output', 'none'])
def test_from_trf(thr, guiding_array):
    """
    Test the creation of ``PureParallel`` out of a transformation
    with various values of the guiding array.
    """

    N = 1000
    coeff = 3
    dtype = numpy.float32

    arr_t = Type(dtype, shape=N)
    trf = mul_param(arr_t, dtype)

    if guiding_array == 'input':
        arr = trf.input
    elif guiding_array == 'output':
        arr = trf.output
    elif guiding_array == 'none':
        arr = None

    p = PureParallel.from_trf(trf, guiding_array=arr)

    # The new PureParallel has to preserve the parameter list of the original transformation.
    assert list(p.signature.parameters.values()) == list(trf.signature.parameters.values())

    a = get_test_array_like(p.parameter.input)
    a_dev = thr.to_device(a)
    res_dev = thr.empty_like(p.parameter.output)

    pc = p.compile(thr)
    pc(res_dev, a_dev, coeff)

    assert diff_is_negligible(res_dev.get(), a * 3)


class SameArgumentHelper(Computation):

    def __init__(self, arr):

        copy_trf = copy(arr, out_arr_t=arr)
        self._copy_comp = PureParallel.from_trf(copy_trf, copy_trf.input)

        Computation.__init__(self, [
            Parameter('outer_output', Annotation(arr, 'o')),
            Parameter('outer_input', Annotation(arr, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()
        temp = plan.temp_array_like(output)
        plan.computation_call(self._copy_comp, temp, input_)
        plan.computation_call(self._copy_comp, temp, temp)
        plan.computation_call(self._copy_comp, output, temp)
        return plan


def test_same_argument(some_thr):
    """
    A regression test for an unexpected interaction of the way PureParallel.from_trf() worked
    and a logic flaw in processing 'io'-type nodes in a transformation tree.

    from_trf() created a trivial computation with a single 'io' parameter and
    attached the given transformation to it.
    This preserved the order of parameters in the resultinc computation
    (because of how the transformation tree was traversed),
    and quite nicely relied only on the public API.

    So, for a simple transformation with one input and one output the root PP computation
    looked like:

        input (io)

    and after attaching the transformation:

        input (i)
        input (o) ---(tr:input -> tr:output)---> output

    When this computation was used inside another computation and was passed the same argument
    (e.g. 'temp') both for input and output, during the translation stage
    this would be transformed to (since 'temp' is passed both to 'input' and 'output')

        temp (i)
        temp (o) ---(tr:input -> tr:output)---> temp

    because the translation was purely name-based.
    This resulted in some cryptic errors due to the name clash.
    Now the masked 'input' should have been mangled instead of translated,
    producing something like

        temp (i)
        _nested_input (o) ---(tr:input -> tr:output)---> temp

    but this functionality was not implemented.
    """
    arr = get_test_array((1000, 8, 1), numpy.complex64)
    arr_dev = some_thr.to_device(arr)

    test = SameArgumentHelper(arr)
    testc = test.compile(some_thr)

    testc(arr_dev, arr_dev)

    assert diff_is_negligible(arr_dev.get(), arr)
