import numpy

from reikna.helpers import template_from, min_blocks
import reikna.cluda.functions as functions
from reikna.core import Computation, Parameter, Annotation, Transformation


# Output = Input * Parameter
def tr_scale(arr, coeff_t):
    return Transformation(
        [Parameter('o1', Annotation(arr, 'o')),
        Parameter('i1', Annotation(arr, 'i')),
        Parameter('s1', Annotation(coeff_t))],
        "${o1.store_same}(${mul}(${i1.load_same}, ${s1}));",
        render_kwds=dict(
            mul=functions.mul(arr.dtype, coeff_t, out_dtype=arr.dtype)))


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
        <%def name="dummy(kernel_declaration, C, D, A, B, coeff)">
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            VSIZE_T idx0 = virtual_global_id(0);
            VSIZE_T idx1 = virtual_global_id(1);

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

        <%def name="dummy2(kernel_declaration, CC, DD, C, D, pers_arr, const_coeff)">
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            VSIZE_T idx0 = virtual_global_id(0);
            VSIZE_T idx1 = virtual_global_id(1);

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

        block_size = 8

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
            test_computation_incorrect_type=False,
            test_same_arg_as_i_and_o=False):

        self._second_coeff = second_coeff
        self._same_A_B = same_A_B
        self._test_same_arg_as_i_and_o = test_same_arg_as_i_and_o

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

        scale = tr_scale(A, numpy.float32)
        nested.parameter.A.connect(scale, scale.o1, A_prime=scale.i1, scale_coeff=scale.s1)

        C_temp = plan.temp_array_like(C)
        D_temp = plan.temp_array_like(D)

        plan.computation_call(
            nested,
            (numpy.empty_like(C_temp) if self._test_computation_adhoc_array else C_temp),
            (B if self._test_computation_incorrect_role else D_temp),
            (B if self._test_computation_incorrect_type else A),
            0.4, B, coeff)

        if self._test_same_arg_as_i_and_o:
            _trash = plan.temp_array_like(C) # ignoring this result
            nested2 = Dummy(A, B, coeff, same_A_B=self._same_A_B)
            plan.computation_call(nested2, _trash, D_temp, C_temp, D_temp, coeff)

        plan.computation_call(nested, C, D, C_temp, 0.4, D_temp, self._second_coeff)

        return plan


def mock_dummy_nested(a, b, coeff, second_coeff, test_same_arg_as_i_and_o=False):
    c, d = mock_dummy(0.4 * a, b, coeff)
    if test_same_arg_as_i_and_o:
        _, d = mock_dummy(c, d, coeff)
    return mock_dummy(0.4 * c, d, second_coeff)


class DummyAdvanced(Computation):
    """
    Dummy computation class which uses some advanced features.
    """

    def __init__(self, arr, coeff):
        Computation.__init__(self, [
            Parameter('C', Annotation(arr, 'io')),
            Parameter('D', Annotation(arr, 'io')),
            Parameter('coeff1', Annotation(coeff)),
            Parameter('coeff2', Annotation(coeff))])

    def _build_plan(self, plan_factory, device_params, C, D, coeff1, coeff2):
        plan = plan_factory()
        nested = Dummy(C, D, coeff1, same_A_B=True)

        C_temp = plan.temp_array_like(C)
        D_temp = plan.temp_array_like(D)

        # Testing a computation call which uses the same argument for two parameters.
        plan.computation_call(nested, C_temp, D, C, C, coeff1)

        arr_dtype = C.dtype
        coeff_dtype = coeff2.dtype

        mul = functions.mul(arr_dtype, coeff_dtype)
        div = functions.div(arr_dtype, coeff_dtype)

        template = template_from("""
        <%def name="dummy(kernel_declaration, CC, C, D, coeff)">
        ${kernel_declaration}
        {
            VIRTUAL_SKIP_THREADS;
            VSIZE_T idx0 = virtual_global_id(0);
            VSIZE_T idx1 = virtual_global_id(1);

            ${CC.store_idx}(idx0, idx1,
                ${C.load_idx}(idx0, idx1) +
                ${mul}(${D.load_idx}(idx0, idx1), ${coeff}));
        }
        </%def>
        """)

        # Testing a kernel call which uses the same argument for two parameters.
        plan.kernel_call(
            template.get_def('dummy'),
            [C, C_temp, C_temp, coeff2],
            global_size=C.shape,
            render_kwds=dict(mul=mul))

        return plan


def mock_dummy_advanced(c, d, coeff1, coeff2):
    ct, d = mock_dummy(c, c, coeff1)
    c = ct + coeff2 * ct
    return c, d
