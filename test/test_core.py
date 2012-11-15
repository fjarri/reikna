import numpy
import pytest

from tigger.helpers import *
from tigger.core import *
from tigger import Transformation, ArrayValue, ScalarValue

from helpers import *


class Dummy(Computation):
    """
    Dummy computation class with two inputs, two outputs and one parameter.
    Will be used to perform core and transformation tests.
    """

    def _get_argnames(self):
        return ('C', 'D'), ('A', 'B'), ('coeff',)

    def _get_basis_for(self, C, D, A, B, coeff):
        assert C.dtype == D.dtype == A.dtype == B.dtype
        return dict(arr_dtype=C.dtype, coeff_dtype=coeff.dtype, size=C.size)

    def _get_argvalues(self, basis):
        av = ArrayValue((basis.size,), basis.arr_dtype)
        sv = ScalarValue(basis.coeff_dtype)
        return dict(C=av, D=av, A=av, B=av, coeff=sv)

    def _construct_operations(self, basis, device_params):
        operations = self._get_operation_recorder()
        template = template_from("""
        <%def name="dummy(C, D, A, B, coeff)">
        ${kernel_definition}
        {
            VIRTUAL_SKIP_THREADS;
            int idx = virtual_global_id(0);
            ${A.ctype} a = ${A.load}(idx);
            ${B.ctype} b = ${B.load}(idx);
            ${C.ctype} c = ${func.mul(A.dtype, coeff.dtype)}(a, ${coeff});
            ${D.ctype} d = ${func.div(B.dtype, coeff.dtype)}(b, ${coeff});
            ${C.store}(idx, c);
            ${D.store}(idx, d);
        }
        </%def>
        """)

        block_size = 128

        operations.add_kernel(
            template, 'dummy',
            ['C', 'D', 'A', 'B', 'coeff'],
            global_size=min_blocks(basis.size, block_size) * block_size,
            local_size=block_size)
        return operations


class DummyNested(Computation):
    """
    Dummy computation class with a nested computation inside.
    """

    def _get_argnames(self):
        return ('C', 'D'), ('A', 'B'), ('coeff',)

    def _get_basis_for(self, C, D, A, B, coeff):
        assert C.dtype == D.dtype == A.dtype == B.dtype
        return dict(arr_dtype=C.dtype, coeff_dtype=coeff.dtype, size=C.size)

    def _get_argvalues(self, basis):
        av = ArrayValue((basis.size,), basis.arr_dtype)
        sv = ScalarValue(basis.coeff_dtype)
        return dict(C=av, D=av, A=av, B=av, coeff=sv)

    def _construct_operations(self, basis, device_params):
        operations = self._get_operation_recorder()
        nested = self.get_nested_computation(Dummy)
        # note that the argument order is changed
        operations.add_computation(nested, 'D', 'C', 'B', 'A', 'coeff')
        return operations


# A function which does the same job as base Dummy kernel
def mock_dummy(a, b, coeff):
    return a * coeff, b / coeff


# Some transformations to use by tests

# Identity transformation: Output = Input
tr_trivial = Transformation(
    inputs=1, outputs=1,
    code="${o1.store}(${i1.load});")

# Output = Input1 * Parameter1 + Input 2
tr_2_to_1 = Transformation(
    inputs=2, outputs=1, scalars=1,
    derive_o_from_is=lambda i1, i2, s1: i1,
    code="""
        ${o1.ctype} t = ${func.mul(o1.dtype, i1.dtype)}(
            ${func.cast(o1.dtype, s1.dtype)}(${s1}), ${i1.load});
        ${o1.store}(t + ${i2.load});
    """)

# Output1 = Input / 2, Output2 = Input / 2
tr_1_to_2 = Transformation(
    inputs=1, outputs=2,
    code="""
        ${o1.ctype} t = ${func.mul(i1.dtype, numpy.float32)}(${i1.load}, 0.5);
        ${o1.store}(t);
        ${o2.store}(t);
    """)

# Output = Input * Parameter
tr_scale = Transformation(
    inputs=1, outputs=1, scalars=1,
    derive_o_from_is=lambda i1, s1: i1,
    derive_i_from_os=lambda o1, s1: o1,
    code="""
        ${o1.store}(
            ${func.mul(i1.dtype, s1.dtype, out=o1.dtype)}(${i1.load}, ${s1})
        );
    """)


def test_non_prepared_call(some_ctx):
    d = Dummy(some_ctx)
    with pytest.raises(InvalidStateError):
        d(None, None, None, None, None)

def test_incorrect_connections(some_ctx):
    d = Dummy(some_ctx)
    d.connect(tr_trivial, 'A', ['A_prime'])
    d.connect(tr_trivial, 'D', ['D_prime'])

    tests = [
        # cannot connect to scalar
        (tr_trivial, 'coeff', ['A_prime']),
        # A is not a leaf anymore, should fail
        (tr_trivial, 'A', ['A_prime']),
        # coeff is an existing scalar node, B is an array
        (tr_trivial, 'B', ['coeff']),
        # second list should contain scalar nodes, but A_prime is an array
        (tr_scale, 'C', ['C_prime'], ['A_prime']),
        # incorrect argument name
        (tr_scale, 'C', ['1C_prime'], ['param']),
        # incorrect argument name
        (tr_scale, 'C', ['C_prime'], ['1param']),
        # Cannot connect output to an existing node.
        # With current limitation of strictly elementwise transformations,
        # connection to an existing output node would cause data loss and is most likely an error.
        # Moreover, with current transformation code generator it creates some complications.
        # (Connection to an existing input or scalar is fine, see corresponding tests)
        (tr_trivial, 'C', ['D']),
        (tr_trivial, 'C', ['D_prime']),
        # incorrect number of inputs/outputs
        (tr_1_to_2, 'A', ['A_prime']),
        (tr_2_to_1, 'C', ['C_prime']),
        (tr_trivial, 'A', ['A_prime', 'B_prime']),
        (tr_trivial, 'C', ['C_prime', 'D_prime']),
        (tr_trivial, 'A', ['A_prime'], ['param'])
    ]

    for test in tests:
        with pytest.raises(ValueError):
            d.connect(*test)

def test_non_array_connection(some_ctx):
    d = Dummy(some_ctx)
    with pytest.raises(ValueError):
        d.connect(tr_trivial, 'coeff', ['A_prime'])

def test_non_existent_connection(some_ctx):
    d = Dummy(some_ctx)
    with pytest.raises(ValueError):
        d.connect(tr_trivial, 'blah', ['A_prime'])

def test_signature_correctness(some_ctx):
    d = Dummy(some_ctx)

    # Signature of non-prepared array: no types, no shapes
    assert d.signature_str() == "(array) C, (array) D, (array) A, (array) B, (scalar) coeff"

    # Connect some transformations and prepare
    d.connect(tr_trivial, 'A', ['A_prime'])
    d.connect(tr_2_to_1, 'B', ['A_prime', 'B_prime'], ['B_param'])
    d.connect(tr_trivial, 'B_prime', ['B_new_prime'])
    d.connect(tr_1_to_2, 'C', ['C_half1', 'C_half2'])
    d.connect(tr_trivial, 'C_half1', ['C_new_half1'])
    d.connect(tr_scale, 'D', ['D_prime'], ['D_param'])

    array = ArrayValue((1024,), numpy.complex64)
    scalar = ScalarValue(numpy.float32)

    d.prepare_for(array, array, array, array, array, scalar, scalar, scalar)

    assert d.signature_str() == (
        "(array, complex64, (1024,)) C_new_half1, "
        "(array, complex64, (1024,)) C_half2, "
        "(array, complex64, (1024,)) D_prime, "
        "(array, complex64, (1024,)) A_prime, "
        "(array, complex64, (1024,)) B_new_prime, "
        "(scalar, float32) coeff, "
        "(scalar, float32) D_param, "
        "(scalar, float32) B_param")

def test_incorrect_number_of_arguments_in_prepare(some_ctx):
    d = Dummy(some_ctx)
    with pytest.raises(TypeError):
        d.prepare_for(None, None, None, None)

def test_incorrect_number_of_arguments_in_call(some_ctx):
    array = ArrayValue((1024,), numpy.complex64)
    scalar = ScalarValue(numpy.float32)

    d = Dummy(some_ctx)
    d.prepare_for(array, array, array, array, scalar)
    with pytest.raises(TypeError):
        d(None, None, None, None)

def test_scalar_instead_of_array(some_ctx):
    N = 1024

    d = Dummy(some_ctx)

    A = get_test_array(N, numpy.complex64)
    B = get_test_array(N, numpy.complex64)
    C = get_test_array(N, numpy.complex64)
    D = get_test_array(N, numpy.complex64)

    with pytest.raises(TypeError):
        d.prepare_for(C, D, A, 2, B)
    with pytest.raises(TypeError):
        d.prepare_for(C, D, A, B, B)

def test_debug_signature_check(some_ctx):
    N1 = 1024
    N2 = 512

    array = ArrayValue(N1, numpy.complex64)
    scalar = ScalarValue(numpy.float32)

    d = Dummy(some_ctx, debug=True)
    d.prepare_for(array, array, array, array, scalar)

    A1 = get_test_array(N1, numpy.complex64)
    B1 = get_test_array(N1, numpy.complex64)
    C1 = get_test_array(N1, numpy.complex64)
    D1 = get_test_array(N1, numpy.complex64)

    A2 = get_test_array(N2, numpy.complex64)
    B2 = get_test_array(N2, numpy.complex64)
    C2 = get_test_array(N2, numpy.complex64)
    D2 = get_test_array(N2, numpy.complex64)

    with pytest.raises(ValueError):
        # this will require basis change
        d(C2, D2, B2, A2, 2)

    with pytest.raises(TypeError):
        # scalar argument in place of array
        d(C1, D1, A1, 2, B1)

    with pytest.raises(TypeError):
        # array argument in place of scalar
        d(C1, D1, A1, B1, B1)

def test_transformations_work(ctx):

    coeff = numpy.float32(2)
    B_param = numpy.float32(3)
    D_param = numpy.float32(4)
    N = 1024

    d = Dummy(ctx)

    d.connect(tr_trivial, 'A', ['A_prime'])
    d.connect(tr_2_to_1, 'B', ['A_prime', 'B_prime'], ['B_param'])
    d.connect(tr_trivial, 'B_prime', ['B_new_prime'])
    d.connect(tr_1_to_2, 'C', ['C_half1', 'C_half2'])
    d.connect(tr_trivial, 'C_half1', ['C_new_half1'])
    d.connect(tr_scale, 'D', ['D_prime'], ['D_param'])

    A_prime = get_test_array(N, numpy.complex64)
    B_new_prime = get_test_array(N, numpy.complex64)
    gpu_A_prime = ctx.to_device(A_prime)
    gpu_B_new_prime = ctx.to_device(B_new_prime)
    gpu_C_new_half1 = ctx.allocate(N, numpy.complex64)
    gpu_C_half2 = ctx.allocate(N, numpy.complex64)
    gpu_D_prime = ctx.allocate(N, numpy.complex64)
    d.prepare_for(
        gpu_C_new_half1, gpu_C_half2, gpu_D_prime,
        gpu_A_prime, gpu_B_new_prime,
        coeff, D_param, B_param)

    d(gpu_C_new_half1, gpu_C_half2, gpu_D_prime,
        gpu_A_prime, gpu_B_new_prime, coeff, D_param, B_param)

    A = A_prime
    B = A_prime * B_param + B_new_prime
    C, D = mock_dummy(A, B, coeff)
    C_new_half1 = C / 2
    C_half2 = C / 2
    D_prime = D * D_param

    assert diff_is_negligible(ctx.from_device(gpu_C_new_half1), C_new_half1)
    assert diff_is_negligible(ctx.from_device(gpu_C_half2), C_half2)
    assert diff_is_negligible(ctx.from_device(gpu_D_prime), D_prime)

def test_connection_to_base(ctx):

    coeff = numpy.float32(2)
    B_param = numpy.float32(3)
    D_param = numpy.float32(4)
    N = 1024

    d = Dummy(ctx)

    # connect to the base array argument (effectively making B the same as A)
    d.connect(tr_trivial, 'A', ['B'])

    # connect to the base scalar argument
    d.connect(tr_scale, 'C', ['C_prime'], ['coeff'])

    B = get_test_array(N, numpy.complex64)
    gpu_B = ctx.to_device(B)
    gpu_C_prime = ctx.allocate(N, numpy.complex64)
    gpu_D = ctx.allocate(N, numpy.complex64)
    d.prepare_for(gpu_C_prime, gpu_D, gpu_B, coeff)
    d(gpu_C_prime, gpu_D, gpu_B, coeff)

    A = B
    C, D = mock_dummy(A, B, coeff)
    C_prime = C * coeff

    assert diff_is_negligible(ctx.from_device(gpu_C_prime), C_prime)
    assert diff_is_negligible(ctx.from_device(gpu_D), D)

def test_nested(ctx):

    coeff = numpy.float32(2)
    B_param = numpy.float32(3)
    D_param = numpy.float32(4)
    N = 1024

    d = DummyNested(ctx)

    d.connect(tr_trivial, 'A', ['A_prime'])
    d.connect(tr_2_to_1, 'B', ['A_prime', 'B_prime'], ['B_param'])
    d.connect(tr_trivial, 'B_prime', ['B_new_prime'])
    d.connect(tr_1_to_2, 'C', ['C_half1', 'C_half2'])
    d.connect(tr_trivial, 'C_half1', ['C_new_half1'])
    d.connect(tr_scale, 'D', ['D_prime'], ['D_param'])

    A_prime = get_test_array(N, numpy.complex64)
    B_new_prime = get_test_array(N, numpy.complex64)
    gpu_A_prime = ctx.to_device(A_prime)
    gpu_B_new_prime = ctx.to_device(B_new_prime)
    gpu_C_new_half1 = ctx.allocate(N, numpy.complex64)
    gpu_C_half2 = ctx.allocate(N, numpy.complex64)
    gpu_D_prime = ctx.allocate(N, numpy.complex64)
    d.prepare_for(
        gpu_C_new_half1, gpu_C_half2, gpu_D_prime,
        gpu_A_prime, gpu_B_new_prime,
        coeff, D_param, B_param)

    d(gpu_C_new_half1, gpu_C_half2, gpu_D_prime,
        gpu_A_prime, gpu_B_new_prime, coeff, D_param, B_param)

    A = A_prime
    B = A_prime * B_param + B_new_prime
    D, C = mock_dummy(B, A, coeff)
    C_new_half1 = C / 2
    C_half2 = C / 2
    D_prime = D * D_param

    assert diff_is_negligible(ctx.from_device(gpu_C_new_half1), C_new_half1)
    assert diff_is_negligible(ctx.from_device(gpu_C_half2), C_half2)
    assert diff_is_negligible(ctx.from_device(gpu_D_prime), D_prime)
