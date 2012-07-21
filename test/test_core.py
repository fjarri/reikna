import numpy

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
        assert len(set([x.dtype for x in (A, B, C, D)])) == 1
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

tr_trivial = Transformation(
    load=1, store=1,
    code="${store.s1}(${load.l1});")

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

tr_1_to_2 = Transformation(
    load=1, store=2,
    code="""
        ${ctype.s1} t = ${func.mul(dtype.l1, numpy.float32)}(${load.l1}, 0.5);
        ${store.s1}(t);
        ${store.s2}(t);
    """)



def test_preprocessing(ctx):

    coeff = numpy.float32(2)
    B_param = numpy.float32(3)
    N = 1024

    d = Dummy(ctx)
    assert d.signature_str() == "(array) C, (array) D, (array) A, (array) B, (scalar) coeff"

    d.connect(tr_trivial, 'A', ['A_prime'])
    d.connect(tr_2_to_1, 'B', ['A_prime', 'B_prime'], ['B_param'])
    d.connect(tr_trivial, 'B_prime', ['B_new_prime'])
    d.connect(tr_1_to_2, 'C', ['C_half1', 'C_half2'])
    d.connect(tr_trivial, 'C_half1', ['C_new_half1'])

    d.prepare(arr_dtype=numpy.float32, size=N)
    assert d.signature_str() == "(array, float32, (1024,)) C_new_half1, " \
        "(array, float32, (1024,)) C_half2, " \
        "(array, float32, (1024,)) D, " \
        "(array, float32, (1024,)) A_prime, (array, float32, (1024,)) B_new_prime, " \
        "(scalar, float32) coeff, (scalar, float32) B_param"

    A_prime = getTestArray(N, numpy.complex64)
    B_new_prime = getTestArray(N, numpy.complex64)
    gpu_A_prime = ctx.to_device(A_prime)
    gpu_B_new_prime = ctx.to_device(B_new_prime)
    gpu_C_new_half1 = ctx.allocate(N, numpy.complex64)
    gpu_C_half2 = ctx.allocate(N, numpy.complex64)
    gpu_D = ctx.allocate(N, numpy.complex64)
    d.prepare_for(gpu_C_new_half1, gpu_C_half2, gpu_D,
        gpu_A_prime, gpu_B_new_prime, numpy.float32(coeff), numpy.int32(B_param))
    assert d.signature_str() == "(array, complex64, (1024,)) C_new_half1, " \
        "(array, complex64, (1024,)) C_half2, (array, complex64, (1024,)) D, " \
        "(array, complex64, (1024,)) A_prime, " \
        "(array, complex64, (1024,)) B_new_prime, (scalar, float32) coeff, " \
        "(scalar, float32) B_param"

    d(gpu_C_new_half1, gpu_C_half2, gpu_D, gpu_A_prime, gpu_B_new_prime, coeff, B_param)

    A = A_prime
    B = A_prime * B_param + B_new_prime
    C, D = mock_dummy(A, B, coeff)
    C_new_half1 = C / 2
    C_half2 = C / 2
    assert diff(ctx.from_device(gpu_C_new_half1), C_new_half1) < SINGLE_EPS
    assert diff(ctx.from_device(gpu_C_half2), C_half2) < SINGLE_EPS
    assert diff(ctx.from_device(gpu_D), D) < SINGLE_EPS


def a_test_():
    # test connection to non-leaf
    # test connection to non-array
    # test connection to non-existing argument
    # test signature correctness
    # test involved transformation tree
    # chech that in debug mode arguments that require different basis cause an error
    # prepare with key not from basis
    # prepare with the same keys as in basis (no way to checkm just for coverage)
    # call with arguments that require different basis (debug should be on)
    # test that the error is thrown when data type conflict occurs during type propagation
    # check store transformation with parameters
    pass
