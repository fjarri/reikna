import numpy
from grunnur import Template, functions

from reikna.core import Annotation, Computation, Parameter, Transformation
from reikna.helpers import min_blocks, product


# Output = Input * Parameter
def tr_scale(arr, coeff_t):
    return Transformation(
        [
            Parameter("o1", Annotation(arr, "o")),
            Parameter("i1", Annotation(arr, "i")),
            Parameter("s1", Annotation(coeff_t)),
        ],
        "${o1.store_same}(${mul}(${i1.load_same}, ${s1}));",
        render_kwds=dict(mul=functions.mul(arr.dtype, coeff_t, out_dtype=arr.dtype)),
    )


class Dummy(Computation):
    """
    Dummy computation class with two inputs, two outputs and one parameter.
    Used to perform core and transformation tests.
    """

    def __init__(
        self,
        arr1,
        arr2,
        coeff,
        *,
        same_a_b=False,
        test_incorrect_parameter_name=False,
        test_untyped_scalar=False,
        test_kernel_adhoc_array=False,
    ):
        assert len(arr1.shape) == 2
        assert len(arr2.shape) == (2 if same_a_b else 1)
        assert arr1.dtype == arr2.dtype
        if same_a_b:
            assert arr1.shape == arr2.shape
        else:
            assert arr1.shape[0] == arr1.shape[1]

        self._same_a_b = same_a_b
        self._persistent_array = (
            numpy.arange(product(arr2.shape)).reshape(arr2.shape).astype(arr2.dtype)
        )

        self._test_untyped_scalar = test_untyped_scalar
        self._test_kernel_adhoc_array = test_kernel_adhoc_array

        Computation.__init__(
            self,
            [
                Parameter(("_C" if test_incorrect_parameter_name else "C"), Annotation(arr1, "o")),
                Parameter("D", Annotation(arr2, "o")),
                Parameter("A", Annotation(arr1, "i")),
                Parameter("B", Annotation(arr2, "i")),
                Parameter("coeff", Annotation(coeff)),
            ],
        )

    def _build_plan(self, plan_factory, _device_params, args):
        plan = plan_factory()

        c = args.C
        d = args.D
        a = args.A
        b = args.B
        coeff = args.coeff

        arr_dtype = c.dtype
        coeff_dtype = coeff.dtype

        mul = functions.mul(arr_dtype, coeff_dtype)
        div = functions.div(arr_dtype, coeff_dtype)

        template = Template.from_string("""
        <%def name="dummy(kernel_declaration, C, D, A, B, coeff)">
        ${kernel_declaration}
        {
            if (${static.skip}()) return;
            VSIZE_T idx0 = ${static.global_id}(0);
            VSIZE_T idx1 = ${static.global_id}(1);

            ${A.ctype} a = ${A.load_idx}(idx0, idx1);
            ${C.ctype} c = ${mul}(a, ${coeff});
            ${C.store_idx}(idx1, idx0, c);

            %if same_a_b:
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
            if (${static.skip}()) return;
            VSIZE_T idx0 = ${static.global_id}(0);
            VSIZE_T idx1 = ${static.global_id}(1);

            ${CC.store_idx}(idx0, idx1, ${C.load_idx}(idx0, idx1));

            %if same_a_b:
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

        c_temp = plan.temp_array_like(c)
        d_temp = plan.temp_array_like(d)
        arr = plan.persistent_array(self._persistent_array)

        plan.kernel_call(
            template.get_def("dummy"),
            [c_temp, d_temp, a, b, coeff],
            global_size=a.shape,
            local_size=(block_size, block_size),
            render_kwds=dict(mul=mul, div=div, same_a_b=self._same_a_b),
        )

        plan.kernel_call(
            template.get_def("dummy2"),
            [
                c,
                d,
                c_temp,
                d_temp,
                (self._persistent_array if self._test_kernel_adhoc_array else arr),
                (10 if self._test_untyped_scalar else numpy.float32(10)),
            ],
            global_size=a.shape,
            local_size=(block_size, block_size),
            render_kwds=dict(mul=mul, same_a_b=self._same_a_b),
        )

        return plan


# A function which does the same job as base Dummy kernel
def mock_dummy(a, b, coeff):
    return a.T * coeff, (b / coeff * 10) + numpy.arange(b.size).reshape(b.shape).astype(b.dtype)


class DummyNested(Computation):
    """Dummy computation class with a nested computation inside."""

    def __init__(
        self,
        arr1,
        arr2,
        coeff,
        second_coeff,
        *,
        same_a_b=False,
        test_computation_adhoc_array=False,
        test_computation_incorrect_role=False,
        test_computation_incorrect_type=False,
        test_same_arg_as_i_and_o=False,
    ):
        self._second_coeff = second_coeff
        self._same_a_b = same_a_b
        self._test_same_arg_as_i_and_o = test_same_arg_as_i_and_o

        self._test_computation_adhoc_array = test_computation_adhoc_array
        self._test_computation_incorrect_role = test_computation_incorrect_role
        self._test_computation_incorrect_type = test_computation_incorrect_type

        Computation.__init__(
            self,
            [
                Parameter("C", Annotation(arr1, "o")),
                Parameter("D", Annotation(arr2, "o")),
                Parameter("A", Annotation(arr1, "i")),
                Parameter("B", Annotation(arr2, "i")),
                Parameter("coeff", Annotation(coeff)),
            ],
        )

    def _build_plan(self, plan_factory, _device_params, args):
        plan = plan_factory()

        c = args.C
        d = args.D
        a = args.A
        b = args.B
        coeff = args.coeff

        nested = Dummy(a, b, coeff, same_a_b=self._same_a_b)

        scale = tr_scale(a, numpy.float32)
        nested.parameter.A.connect(scale, scale.o1, A_prime=scale.i1, scale_coeff=scale.s1)

        c_temp = plan.temp_array_like(c)
        d_temp = plan.temp_array_like(d)

        plan.computation_call(
            nested,
            (numpy.empty_like(c_temp) if self._test_computation_adhoc_array else c_temp),
            (b if self._test_computation_incorrect_role else d_temp),
            (b if self._test_computation_incorrect_type else a),
            0.4,
            b,
            coeff,
        )

        if self._test_same_arg_as_i_and_o:
            _trash = plan.temp_array_like(c)  # ignoring this result
            nested2 = Dummy(a, b, coeff, same_a_b=self._same_a_b)
            plan.computation_call(nested2, _trash, d_temp, c_temp, d_temp, coeff)

        plan.computation_call(nested, c, d, c_temp, 0.4, d_temp, self._second_coeff)

        return plan


def mock_dummy_nested(a, b, coeff, second_coeff, *, test_same_arg_as_i_and_o=False):
    c, d = mock_dummy(0.4 * a, b, coeff)
    if test_same_arg_as_i_and_o:
        _, d = mock_dummy(c, d, coeff)
    return mock_dummy(0.4 * c, d, second_coeff)


class DummyAdvanced(Computation):
    """Dummy computation class which uses some advanced features."""

    def __init__(self, arr, coeff):
        Computation.__init__(
            self,
            [
                Parameter("C", Annotation(arr, "io")),
                Parameter("D", Annotation(arr, "io")),
                Parameter("coeff1", Annotation(coeff)),
                Parameter("coeff2", Annotation(coeff)),
            ],
        )

    def _build_plan(self, plan_factory, _device_params, args):
        plan = plan_factory()

        c = args.C
        d = args.D
        coeff1 = args.coeff1
        coeff2 = args.coeff2

        nested = Dummy(c, d, coeff1, same_a_b=True)

        c_temp = plan.temp_array_like(c)

        # Testing a computation call which uses the same argument for two parameters.
        plan.computation_call(nested, c_temp, d, c, c, coeff1)

        arr_dtype = c.dtype
        coeff_dtype = coeff2.dtype

        mul = functions.mul(arr_dtype, coeff_dtype)

        template = Template.from_string("""
        <%def name="dummy(kernel_declaration, CC, C, D, coeff)">
        ${kernel_declaration}
        {
            if (${static.skip}()) return;
            VSIZE_T idx0 = ${static.global_id}(0);
            VSIZE_T idx1 = ${static.global_id}(1);

            ${CC.store_idx}(idx0, idx1,
                ${C.load_idx}(idx0, idx1) +
                ${mul}(${D.load_idx}(idx0, idx1), ${coeff}));
        }
        </%def>
        """)

        # Testing a kernel call which uses the same argument for two parameters.
        plan.kernel_call(
            template.get_def("dummy"),
            [c, c_temp, c_temp, coeff2],
            global_size=c.shape,
            render_kwds=dict(mul=mul),
        )

        return plan


def mock_dummy_advanced(c, d, coeff1, coeff2):
    ct, d = mock_dummy(c, c, coeff1)
    c = ct + coeff2 * ct
    return c, d
