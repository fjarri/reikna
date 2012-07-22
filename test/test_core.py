import numpy
import pytest

from tigger.core import *
from tigger import Transformation

from helpers import *


class Dummy(Computation):
    """
    Dummy computation class with two inputs, two outputs and one parameter.
    Will be used to perform core and transformation tests.
    """

    def _get_default_basis(self):
        return dict(arr_dtype=numpy.float32, coeff_dtype=numpy.float32, size=1)

    def _construct_basis(self, C, D, A, B, coeff):
        return dict(arr_dtype=C.dtype, coeff_dtype=coeff.dtype, size=C.size)

    def _get_base_signature(self):
        bs = self._basis
        av = ArrayValue((bs.size,), bs.arr_dtype)
        sv = ScalarValue(None, bs.coeff_dtype)
        return (
            [('C', av), ('D', av)],
            [('A', av), ('B', av)],
            [('coeff', sv)])

    def _construct_kernels(self):
        # basis will be passed automatically as a keyword
        # optional keywords can be passed here
        # TODO: is it a good way to specify templates?
        bs = self._basis

        template = template_from("""
        KERNEL void dummy(${signature})
        {
            int idx = LID_0 + LSIZE_0 * GID_0;
            if (idx < ${size})
            {
                ${ctype.A} a = ${load.A}(idx);
                ${ctype.B} b = ${load.B}(idx);
                ${ctype.C} c = ${func.mul(dtype.A, dtype.coeff)}(a, ${param.coeff});
                ${ctype.D} d = ${func.div(dtype.B, dtype.coeff)}(b, ${param.coeff});
                ${store.C}(idx, c);
                ${store.D}(idx, d);
            }
        }
        """)

        src = self._render(template, size=bs.size)
        block_size = 128

        return [KernelCall(
            'dummy',
            ['C', 'D', 'A', 'B', 'coeff'],
            src,
            grid=(block_size, 1),
            block=(min_blocks(bs.size, block_size), 1, 1)
        )]


# A function which does the same job as base Dummy kernel
def mock_dummy(a, b, coeff):
    return a * coeff, b / coeff


# Some transformations to use by tests

# Identity transformation: Output = Input
tr_trivial = Transformation(
    load=1, store=1,
    code="${store.s1}(${load.l1});")

# Output = Input1 * Parameter1 + Input 2
tr_2_to_1 = Transformation(
    load=2, store=1, parameters=1,
    derive_s_from_lp=lambda l1, l2, p1: [l1],
    derive_lp_from_s=lambda s1: ([s1, s1], [numpy.float32]),
    derive_l_from_sp=lambda s1, p1: [s1, s1],
    derive_sp_from_l=lambda l1, l2: ([l1], [numpy.float32]),
    code="""
        ${ctype.s1} t = ${func.mul(dtype.s1, dtype.l1)}(
            ${func.cast(dtype.s1, dtype.p1)}(${param.p1}), ${load.l1});
        ${store.s1}(t + ${load.l2});
    """)

# Output1 = Input / 2, Output2 = Input / 2
tr_1_to_2 = Transformation(
    load=1, store=2,
    code="""
        ${ctype.s1} t = ${func.mul(dtype.l1, numpy.float32)}(${load.l1}, 0.5);
        ${store.s1}(t);
        ${store.s2}(t);
    """)

# Output = Input * Parameter
tr_scale = Transformation(
    load=1, store=1, parameters=1,
    derive_s_from_lp=lambda l1, p1: [l1],
    derive_lp_from_s=lambda s1: ([s1], [numpy.float32]),
    derive_l_from_sp=lambda s1, p1: [s1],
    derive_sp_from_l=lambda l1: ([l1], [numpy.float32]),
    code="""
        ${store.s1}(
            ${func.mul(dtype.l1, dtype.p1, out=dtype.s1)}(${load.l1}, ${param.p1})
        );
    """)


def test_non_prepared_call(some_ctx):
    d = Dummy(some_ctx)
    with pytest.raises(NotPreparedError):
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
        (tr_trivial, 'C', ['D_prime'])
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
    d.prepare(arr_dtype=numpy.complex64, size=1024)
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
    d = Dummy(some_ctx)
    d.prepare(arr_dtype=numpy.complex64, size=1024)
    with pytest.raises(TypeError):
        d(None, None, None, None)

def test_scalar_instead_of_array(some_ctx):
    N = 1024

    d = Dummy(some_ctx)

    A = getTestArray(N, numpy.complex64)
    B = getTestArray(N, numpy.complex64)
    C = getTestArray(N, numpy.complex64)
    D = getTestArray(N, numpy.complex64)

    with pytest.raises(TypeError):
        d.prepare_for(C, D, A, 2, B)
    with pytest.raises(TypeError):
        d.prepare_for(C, D, A, B, B)

def test_debug_signature_check(some_ctx):
    N1 = 1024
    N2 = 512

    d = Dummy(some_ctx, debug=True)
    d.prepare(arr_dtype=numpy.complex64, size=N1)

    A1 = getTestArray(N1, numpy.complex64)
    B1 = getTestArray(N1, numpy.complex64)
    C1 = getTestArray(N1, numpy.complex64)
    D1 = getTestArray(N1, numpy.complex64)

    A2 = getTestArray(N2, numpy.complex64)
    B2 = getTestArray(N2, numpy.complex64)
    C2 = getTestArray(N2, numpy.complex64)
    D2 = getTestArray(N2, numpy.complex64)

    with pytest.raises(ValueError):
        # this will require basis change
        d(C2, D2, B2, A2, 2)

    with pytest.raises(TypeError):
        # scalar argument in place of array
        d(C1, D1, A1, 2, B1)

    with pytest.raises(TypeError):
        # array argument in place of scalar
        d(C1, D1, A1, B1, B1)

def test_prepare_unknown_key(some_ctx):
    d = Dummy(some_ctx)
    with pytest.raises(KeyError):
        d.prepare(unknown_key=1)

def test_prepare_same_keys(some_ctx):
    d = Dummy(some_ctx)
    kwds = dict(arr_dtype=numpy.complex64, size=1024)

    d.prepare(**kwds)

    # Prepare second time with the same keywords.
    # There is no way to check that the kernel building stage is skipped in this case,
    # so we're doing it only for the sake of coverage.
    d.prepare(**kwds)

def test_type_propagation_conflict(some_ctx):

    # This transformation connects an external complex input
    # to an internal real input (by discarding the complex part)
    tr_complex_to_real = Transformation(
        load=1, store=1,
        derive_s_from_lp=lambda l1, _: [dtypes.real_for(l1)],
        derive_lp_from_s=lambda s1: ([dtypes.complex_for(s1)], []),
        code="${store.s1}((${load.l1}).x);")

    d = Dummy(some_ctx)

    # Both tr_complex_to_real and tr_trivial are perfectly valid transformtions.
    # But if we connect them to lead to the same external variable,
    # there will be a conflict during type derivation:
    # tr_complex_to_real will need the external variable to be complex,
    # and tr_trivial will need it to be real.
    d.connect(tr_complex_to_real, 'A', ['A_new'])
    d.connect(tr_trivial, 'B', ['A_new'])

    with pytest.raises(TypePropagationError):
        d.prepare(arr_dtype=numpy.float32, size=1024)

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

    A_prime = getTestArray(N, numpy.complex64)
    B_new_prime = getTestArray(N, numpy.complex64)
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

    assert diff(ctx.from_device(gpu_C_new_half1), C_new_half1) < SINGLE_EPS
    assert diff(ctx.from_device(gpu_C_half2), C_half2) < SINGLE_EPS
    assert diff(ctx.from_device(gpu_D_prime), D_prime) < SINGLE_EPS

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
    print d.signature_str()

    B = getTestArray(N, numpy.complex64)
    gpu_B = ctx.to_device(B)
    gpu_C_prime = ctx.allocate(N, numpy.complex64)
    gpu_D = ctx.allocate(N, numpy.complex64)
    d.prepare_for(gpu_C_prime, gpu_D, gpu_B, coeff)
    d(gpu_C_prime, gpu_D, gpu_B, coeff)

    A = B
    C, D = mock_dummy(A, B, coeff)
    C_prime = C * coeff

    assert diff(ctx.from_device(gpu_C_prime), C_prime) < SINGLE_EPS
    assert diff(ctx.from_device(gpu_D), D) < SINGLE_EPS
