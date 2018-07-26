import pytest

from reikna.helpers import template_from
from reikna.core import Computation, Parameter, Annotation, Transformation, Type

from helpers import *


class Dummy(Computation):
    """
    Dummy computation class with two inputs, two outputs and one parameter.
    Used to perform core and transformation tests.
    """

    def __init__(self, length, arr1, arr2):
        assert arr1.shape == (length,)
        assert arr2.shape == (2, length)
        self._arr1 = arr1
        self._arr2 = arr2
        Computation.__init__(self, [
            Parameter('output', Annotation(Type(numpy.float32, length), 'o')),
            ])

    def _build_plan(self, plan_factory, device_params, output):
        plan = plan_factory()

        template = template_from("""
        <%def name="dummy(kernel_declaration, output, arr1, arr2)">
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            const VSIZE_T i = virtual_global_id(0);
            ${arr1.ctype} x1 = ${arr1.load_idx}(i);
            ${arr2.ctype} x2 = ${arr2.load_idx}(0, i);
            ${arr2.ctype} x3 = ${arr2.load_idx}(1, i);
            ${output.store_idx}(i, (x2 + x3) * x1);
        }
        </%def>
        """)

        arr1 = plan.constant_array(self._arr1)
        arr2 = plan.constant_array(self._arr2)

        plan.kernel_call(
            template.get_def('dummy'),
            [output, arr1, arr2],
            global_size=output.shape)

        return plan


class DummyOuter(Computation):

    def __init__(self, length, arr1, arr2):
        assert arr1.shape == (length,)
        assert arr2.shape == (2, length)
        self._arr1 = arr1
        self._arr2 = arr2
        Computation.__init__(self, [
            Parameter('output', Annotation(Type(numpy.float32, length), 'o')),
            ])

    def _build_plan(self, plan_factory, device_params, output):
        plan = plan_factory()
        dummy = Dummy(self._arr1.shape[0], self._arr1, self._arr2)
        plan.computation_call(dummy, output)
        return plan


def test_constant_arrays_computation(thr):

    N = 200
    arr1 = get_test_array(N, numpy.int32)
    arr2 = get_test_array((2, N), numpy.float32)
    ref = (arr1 * (arr2[0] + arr2[1])).astype(numpy.float32)

    d = Dummy(N, arr1, arr2).compile(thr)
    out_dev = thr.empty_like(d.parameter.output)
    d(out_dev)
    test = out_dev.get()

    assert diff_is_negligible(test, ref)


def test_constant_arrays_computation_nested(thr):
    """
    Check that constant arrays from a nested computation are
    transfered to the outer computation.
    """

    N = 200
    arr1 = get_test_array(N, numpy.int32)
    arr2 = get_test_array((2, N), numpy.float32)
    ref = (arr1 * (arr2[0] + arr2[1])).astype(numpy.float32)

    d = DummyOuter(N, arr1, arr2).compile(thr)
    out_dev = thr.empty_like(d.parameter.output)
    d(out_dev)
    test = out_dev.get()

    assert diff_is_negligible(test, ref)
