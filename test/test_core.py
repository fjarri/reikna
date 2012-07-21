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
        return (
            [
                ('C', ArrayValue((bs.size,), bs.arr_dtype)),
                ('D', ArrayValue((bs.size,), bs.arr_dtype))
            ],
            [
                ('A', ArrayValue((bs.size,), bs.arr_dtype)),
                ('B', ArrayValue((bs.size,), bs.arr_dtype))
            ],
            [('coeff', ScalarValue(None, bs.coeff_dtype))])

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

def test_non_prepared_call(some_ctx):
    d = Dummy(some_ctx)
    with pytest.raises(NotPreparedError):
        d(None, None, None, None, None)

def test_non_leaf_connection(some_ctx):
    d = Dummy(some_ctx)
    d.connect(tr_trivial, 'A', ['A_prime'])
    with pytest.raises(ValueError):
        d.connect(tr_trivial, 'A', ['A_prime'])

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
    d.prepare(arr_dtype=numpy.complex64, size=1024)
    assert d.signature_str() == (
        "(array, complex64, (1024,)) C_new_half1, "
        "(array, complex64, (1024,)) C_half2, "
        "(array, complex64, (1024,)) D, "
        "(array, complex64, (1024,)) A_prime, "
        "(array, complex64, (1024,)) B_new_prime, "
        "(scalar, float32) coeff, "
        "(scalar, float32) B_param")

def test_transformations_work(ctx):

    coeff = numpy.float32(2)
    B_param = numpy.float32(3)
    N = 1024

    d = Dummy(ctx)

    d.connect(tr_trivial, 'A', ['A_prime'])
    d.connect(tr_2_to_1, 'B', ['A_prime', 'B_prime'], ['B_param'])
    d.connect(tr_trivial, 'B_prime', ['B_new_prime'])
    d.connect(tr_1_to_2, 'C', ['C_half1', 'C_half2'])
    d.connect(tr_trivial, 'C_half1', ['C_new_half1'])

    A_prime = getTestArray(N, numpy.complex64)
    B_new_prime = getTestArray(N, numpy.complex64)
    gpu_A_prime = ctx.to_device(A_prime)
    gpu_B_new_prime = ctx.to_device(B_new_prime)
    gpu_C_new_half1 = ctx.allocate(N, numpy.complex64)
    gpu_C_half2 = ctx.allocate(N, numpy.complex64)
    gpu_D = ctx.allocate(N, numpy.complex64)
    d.prepare_for(gpu_C_new_half1, gpu_C_half2, gpu_D,
        gpu_A_prime, gpu_B_new_prime, numpy.float32(coeff), numpy.int32(B_param))

    d(gpu_C_new_half1, gpu_C_half2, gpu_D, gpu_A_prime, gpu_B_new_prime, coeff, B_param)

    A = A_prime
    B = A_prime * B_param + B_new_prime
    C, D = mock_dummy(A, B, coeff)
    C_new_half1 = C / 2
    C_half2 = C / 2
    assert diff(ctx.from_device(gpu_C_new_half1), C_new_half1) < SINGLE_EPS
    assert diff(ctx.from_device(gpu_C_half2), C_half2) < SINGLE_EPS
    assert diff(ctx.from_device(gpu_D), D) < SINGLE_EPS

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

# prepare with key not from basis
# prepare with the same keys as in basis (no way to checkm just for coverage)
# test that the error is thrown when data type conflict occurs during type propagation
# check store transformation with parameters
