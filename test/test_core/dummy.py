import numpy

from reikna.helpers import template_from, min_blocks
import reikna.cluda.functions as functions
from reikna.core import Computation, Parameter, Annotation


class Dummy(Computation):
    """
    Dummy computation class with two inputs, two outputs and one parameter.
    Used to perform core and transformation tests.
    """

    def __init__(self, arr1, arr2, coeff, same_A_B=False,
            test_incorrect_parameter_name=False,
            test_untyped_scalar=False,
            test_kernel_adhoc_array=False):

        assert len(arr1.shape) == 2
        assert len(arr2.shape) == (2 if same_A_B else 1)
        assert arr1.dtype == arr2.dtype
        if same_A_B:
            assert arr1.shape == arr2.shape
        else:
            assert arr1.shape[0] == arr1.shape[1]

        self._same_A_B = same_A_B
        self._persistent_array = numpy.arange(arr2.size).reshape(arr2.shape).astype(arr2.dtype)

        self._test_untyped_scalar = test_untyped_scalar
        self._test_kernel_adhoc_array = test_kernel_adhoc_array

        Computation.__init__(self, [
            Parameter(('_C' if test_incorrect_parameter_name else 'C'), Annotation(arr1, 'o')),
            Parameter('D', Annotation(arr2, 'o')),
            Parameter('A', Annotation(arr1, 'i')),
            Parameter('B', Annotation(arr2, 'i')),
            Parameter('coeff', Annotation(coeff))])

    def _build_plan(self, plan_factory, device_params, C, D, A, B, coeff):
        plan = plan_factory()

        arr_dtype = C.dtype
        coeff_dtype = coeff.dtype

        mul = functions.mul(arr_dtype, coeff_dtype)
        div = functions.div(arr_dtype, coeff_dtype)

        template = template_from("""
        <%def name="dummy(C, D, A, B, coeff)">
        ${kernel_definition}
        {
            VIRTUAL_SKIP_THREADS;
            int idx0 = virtual_global_id(1);
            int idx1 = virtual_global_id(0);

            ${A.ctype} a = ${A.load_idx}(idx0, idx1);
            ${C.ctype} c = ${mul}(a, ${coeff});
            ${C.store_idx}(idx1, idx0, c);

            %if same_A_B:
                ${B.ctype} b = ${B.load_idx}(idx0, idx1);
                ${D.ctype} d = ${div}(b, ${coeff});
                ${D.store_idx}(idx0, idx1, d);
            %else:
            if (idx1 == 0)
            {
                ${B.ctype} b = ${B.load_idx}(idx0);
                ${D.ctype} d = ${div}(b, ${coeff});
                ${D.store_idx}(idx0, d);
            }
            %endif
        }
        </%def>

        <%def name="dummy2(CC, DD, C, D, pers_arr, const_coeff)">
        ${kernel_definition}
        {
            VIRTUAL_SKIP_THREADS;
            int idx0 = virtual_global_id(1);
            int idx1 = virtual_global_id(0);

            ${CC.store_idx}(idx0, idx1, ${C.load_idx}(idx0, idx1));

            %if same_A_B:
                ${DD.store_idx}(
                    idx0, idx1,
                    ${mul}(${D.load_idx}(idx0, idx1), ${const_coeff}) +
                        ${pers_arr.load_idx}(idx0, idx1));
            %else:
            if (idx1 == 0)
            {
                ${DD.store_idx}(
                    idx0,
                    ${mul}(${D.load_idx}(idx0), ${const_coeff}) +
                        ${pers_arr.load_idx}(idx0));
            }
            %endif
        }
        </%def>
        """)

        block_size = 16

        C_temp = plan.temp_array_like(C)
        D_temp = plan.temp_array_like(D)
        arr = plan.persistent_array(self._persistent_array)

        plan.kernel_call(
            template.get_def('dummy'),
            [C_temp, D_temp, A, B, coeff],
            global_size=A.shape,
            local_size=(block_size, block_size),
            render_kwds=dict(mul=mul, div=div, same_A_B=self._same_A_B))

        plan.kernel_call(
            template.get_def('dummy2'),
            [C, D, C_temp, D_temp,
                (self._persistent_array if self._test_kernel_adhoc_array else arr),
                (10 if self._test_untyped_scalar else numpy.float32(10))],
            global_size=A.shape,
            local_size=(block_size, block_size),
            render_kwds=dict(mul=mul, same_A_B=self._same_A_B))

        return plan


# A function which does the same job as base Dummy kernel
def mock_dummy(a, b, coeff):
    return a.T * coeff, (b / coeff * 10) + numpy.arange(b.size).reshape(b.shape).astype(b.dtype)


class DummyNested(Computation):
    """
    Dummy computation class with a nested computation inside.
    """

    def __init__(self, arr1, arr2, coeff, second_coeff, same_A_B=False,
            test_computation_adhoc_array=False,
            test_computation_incorrect_role=False,
            test_computation_incorrect_type=False):

        self._second_coeff = second_coeff
        self._same_A_B = same_A_B

        self._test_computation_adhoc_array = test_computation_adhoc_array
        self._test_computation_incorrect_role = test_computation_incorrect_role
        self._test_computation_incorrect_type = test_computation_incorrect_type

        Computation.__init__(self, [
            Parameter('C', Annotation(arr1, 'o')),
            Parameter('D', Annotation(arr2, 'o')),
            Parameter('A', Annotation(arr1, 'i')),
            Parameter('B', Annotation(arr2, 'i')),
            Parameter('coeff', Annotation(coeff))])

    def _build_plan(self, plan_factory, device_params, C, D, A, B, coeff):
        plan = plan_factory()
        nested = Dummy(A, B, coeff, same_A_B=self._same_A_B)

        C_temp = plan.temp_array_like(C)
        D_temp = plan.temp_array_like(D)

        plan.computation_call(
            nested,
            (numpy.empty_like(C_temp) if self._test_computation_adhoc_array else C_temp),
            (B if self._test_computation_incorrect_role else D_temp),
            (B if self._test_computation_incorrect_type else A),
            B, coeff)
        plan.computation_call(nested, C, D, C_temp, D_temp, self._second_coeff)

        return plan
