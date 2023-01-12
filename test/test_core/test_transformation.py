import numpy
import pytest

from grunnur import functions, Array

from reikna.core import Parameter, Annotation, Transformation
from reikna.core.signature import Type
from reikna.algorithms import PureParallel
from reikna import transformations

from helpers import *
from test_core.dummy import *


# Some transformations to use by tests

# Identity transformation: Output = Input
def tr_identity(arr):
    return Transformation(
        [Parameter('o1', Annotation(arr, 'o')),
        Parameter('i1', Annotation(arr, 'i'))],
        "${o1.store_same}(${i1.load_same});")

# Output = Input1 * Parameter1 + Input 2
def tr_2_to_1(arr, scalar):
    return Transformation(
        [Parameter('o1', Annotation(arr, 'o')),
        Parameter('i1', Annotation(arr, 'i')),
        Parameter('i2', Annotation(arr, 'i')),
        Parameter('s1', Annotation(scalar))],
        """
        ${o1.ctype} t = ${mul}(${cast}(${s1}), ${i1.load_same});
        ${o1.store_same}(t + ${i2.load_same});
        """,
        render_kwds= dict(
            mul=functions.mul(arr.dtype, arr.dtype),
            cast=functions.cast(scalar.dtype, arr.dtype)))

# Output1 = Input / 2, Output2 = Input / 2
def tr_1_to_2(arr):
    return Transformation(
        [Parameter('o1', Annotation(arr, 'o')),
        Parameter('o2', Annotation(arr, 'o')),
        Parameter('i1', Annotation(arr, 'i'))],
        """
        ${o1.ctype} t = ${mul}(${i1.load_same}, 0.5);
        ${o1.store_same}(t);
        ${o2.store_same}(t);
        """,
        render_kwds=dict(
            mul=functions.mul(arr.dtype, numpy.float32)))


def test_io_parameter_in_transformation():
    with pytest.raises(ValueError):
        tr = Transformation(
            [Parameter('o1', Annotation(Type(numpy.float32, shape=100), 'io'))],
            "${o1.store_same}(${o1.load_same});")


def test_signature_correctness():

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)

    # Root signature
    assert list(d.signature.parameters.values()) == [
        Parameter('C', Annotation(arr_type, 'o')),
        Parameter('D', Annotation(arr_type, 'o')),
        Parameter('A', Annotation(arr_type, 'i')),
        Parameter('B', Annotation(arr_type, 'i')),
        Parameter('coeff', Annotation(coeff_dtype))]

    identity = tr_identity(d.parameter.A)
    join = tr_2_to_1(d.parameter.A, d.parameter.coeff)
    split = tr_1_to_2(d.parameter.A)
    scale = tr_scale(d.parameter.A, d.parameter.coeff.dtype)

    # Connect some transformations
    d.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)
    d.parameter.B.connect(join, join.o1, A_prime=join.i1, B_prime=join.i2, B_param=join.s1)
    d.parameter.B_prime.connect(identity, identity.o1, B_new_prime=identity.i1)
    d.parameter.C.connect(split, split.i1, C_half1=split.o1, C_half2=split.o2)
    d.parameter.C_half1.connect(identity, identity.i1, C_new_half1=identity.o1)
    d.parameter.D.connect(scale, scale.i1, D_prime=scale.o1, D_param=scale.s1)

    assert list(d.signature.parameters.values()) == [
        Parameter('C_new_half1', Annotation(arr_type, 'o')),
        Parameter('C_half2', Annotation(arr_type, 'o')),
        Parameter('D_prime', Annotation(arr_type, 'o')),
        Parameter('D_param', Annotation(coeff_dtype)),
        Parameter('A_prime', Annotation(arr_type, 'i')),
        Parameter('B_new_prime', Annotation(arr_type, 'i')),
        Parameter('B_param', Annotation(coeff_dtype)),
        Parameter('coeff', Annotation(coeff_dtype))]


def test_same_shape(queue):

    N = 200
    coeff = 2
    B_param = 3
    D_param = 4

    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)

    identity = tr_identity(d.parameter.A)
    join = tr_2_to_1(d.parameter.A, d.parameter.coeff)
    split = tr_1_to_2(d.parameter.A)
    scale = tr_scale(d.parameter.A, d.parameter.coeff.dtype)

    d.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)
    d.parameter.B.connect(join, join.o1, A_prime=join.i1, B_prime=join.i2, B_param=join.s1)
    d.parameter.B_prime.connect(identity, identity.o1, B_new_prime=identity.i1)
    d.parameter.C.connect(split, split.i1, C_half1=split.o1, C_half2=split.o2)
    d.parameter.C_half1.connect(identity, identity.i1, C_new_half1=identity.o1)
    d.parameter.D.connect(scale, scale.i1, D_prime=scale.o1, D_param=scale.s1)
    dc = d.compile(queue.device)

    A_prime = get_test_array_like(d.parameter.A_prime)
    B_new_prime = get_test_array_like(d.parameter.B_new_prime)

    A_prime_dev = Array.from_host(queue, A_prime)
    B_new_prime_dev = Array.from_host(queue, B_new_prime)
    C_new_half1_dev = Array.empty_like(queue.device, d.parameter.A_prime)
    C_half2_dev = Array.empty_like(queue.device, d.parameter.A_prime)
    D_prime_dev = Array.empty_like(queue.device, d.parameter.A_prime)

    dc(
        queue,
        C_new_half1_dev, C_half2_dev,
        D_prime_dev, D_param,
        A_prime_dev, B_new_prime_dev,
        B_param, coeff)

    A = A_prime
    B = A_prime * B_param + B_new_prime
    C, D = mock_dummy(A, B, coeff)
    C_new_half1 = C / 2
    C_half2 = C / 2
    D_prime = D * D_param

    assert diff_is_negligible(C_new_half1_dev.get(queue), C_new_half1)
    assert diff_is_negligible(C_half2_dev.get(queue), C_half2)
    assert diff_is_negligible(D_prime_dev.get(queue), D_prime)


def test_connection_to_base(queue):

    N = 200
    coeff = 2

    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)

    identity = tr_identity(d.parameter.A)
    scale = tr_scale(d.parameter.A, d.parameter.coeff.dtype)

    # connect to the base array argument (effectively making B the same as A)
    d.parameter.A.connect(identity, identity.o1, B=identity.i1)
    # connect to the base scalar argument
    d.parameter.C.connect(scale, scale.i1, C_prime=scale.o1, coeff=scale.s1)

    assert list(d.signature.parameters.values()) == [
        Parameter('C_prime', Annotation(arr_type, 'o')),
        Parameter('D', Annotation(arr_type, 'o')),
        Parameter('B', Annotation(arr_type, 'i')),
        Parameter('coeff', Annotation(coeff_dtype))]

    dc = d.compile(queue.device)

    B = get_test_array_like(d.parameter.B)
    B_dev = Array.from_host(queue, B)
    C_prime_dev = Array.empty_like(queue.device, d.parameter.B)
    D_dev = Array.empty_like(queue.device, d.parameter.B)

    dc(queue, C_prime_dev, D_dev, B_dev, coeff)

    C, D = mock_dummy(B, B, coeff)
    C_prime = C * coeff

    assert diff_is_negligible(C_prime_dev.get(queue), C_prime)
    assert diff_is_negligible(D_dev.get(queue), D)


def test_nested_same_shape(queue):

    N = 2000
    coeff = 2
    second_coeff = 7
    B_param = 3
    D_param = 4

    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = DummyNested(arr_type, arr_type, coeff_dtype, second_coeff, same_A_B=True)

    identity = tr_identity(d.parameter.A)
    join = tr_2_to_1(d.parameter.A, d.parameter.coeff)
    split = tr_1_to_2(d.parameter.A)
    scale = tr_scale(d.parameter.A, d.parameter.coeff.dtype)

    d.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)
    d.parameter.B.connect(join, join.o1, A_prime=join.i1, B_prime=join.i2, B_param=join.s1)
    d.parameter.B_prime.connect(identity, identity.o1, B_new_prime=identity.i1)
    d.parameter.C.connect(split, split.i1, C_half1=split.o1, C_half2=split.o2)
    d.parameter.C_half1.connect(identity, identity.i1, C_new_half1=identity.o1)
    d.parameter.D.connect(scale, scale.i1, D_prime=scale.o1, D_param=scale.s1)
    dc = d.compile(queue.device)

    A_prime = get_test_array_like(d.parameter.A_prime)
    B_new_prime = get_test_array_like(d.parameter.B_new_prime)

    A_prime_dev = Array.from_host(queue, A_prime)
    B_new_prime_dev = Array.from_host(queue, B_new_prime)
    C_new_half1_dev = Array.empty_like(queue.device, d.parameter.A_prime)
    C_half2_dev = Array.empty_like(queue.device, d.parameter.A_prime)
    D_prime_dev = Array.empty_like(queue.device, d.parameter.A_prime)

    dc(
        queue,
        C_new_half1_dev, C_half2_dev,
        D_prime_dev, D_param,
        A_prime_dev, B_new_prime_dev,
        B_param, coeff)

    A = A_prime
    B = A_prime * B_param + B_new_prime
    C, D = mock_dummy_nested(A, B, coeff, second_coeff)
    C_new_half1 = C / 2
    C_half2 = C / 2
    D_prime = D * D_param

    assert diff_is_negligible(C_new_half1_dev.get(queue), C_new_half1)
    assert diff_is_negligible(C_half2_dev.get(queue), C_half2)
    assert diff_is_negligible(D_prime_dev.get(queue), D_prime)


def test_strings_as_parameters():
    """
    Check that one can connect transformations using strings as identifiers
    for computation and transformation parameters.
    """

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)
    identity = tr_identity(d.parameter.A)

    d.connect('A', identity, 'o1', A_prime='i1')

    assert list(d.signature.parameters.values()) == [
        Parameter('C', Annotation(arr_type, 'o')),
        Parameter('D', Annotation(arr_type, 'o')),
        Parameter('A_prime', Annotation(arr_type, 'i')),
        Parameter('B', Annotation(arr_type, 'i')),
        Parameter('coeff', Annotation(coeff_dtype))]


def test_alien_parameters():
    """
    Check that one cannot connect transformations using parameter objects from
    other transformation/computation.
    """

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)
    d2 = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)
    identity = tr_identity(d.parameter.A)
    identity2 = tr_identity(d.parameter.A)

    with pytest.raises(ValueError):
        d.connect(d2.parameter.A, identity, 'o1', A_prime='i1')

    with pytest.raises(ValueError):
        d.connect(d.parameter.A, identity, identity2.o1, A_prime='i1')

    with pytest.raises(ValueError):
        d.parameter.A.connect(identity, identity.o1, A_prime=identity2.i1)


def test_connector_repetition():
    """Check that the connector id cannot be repeated in the connections list."""

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)
    identity = tr_identity(d.parameter.A)

    with pytest.raises(ValueError):
        d.parameter.A.connect(identity, identity.o1, A=identity.o1, A_prime=identity.i1)


def test_wrong_connector():
    """Check that the error is thrown if the connector is unknown or is not in the signature."""

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)
    identity = tr_identity(d.parameter.A)

    d.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)

    # Connector is missing
    with pytest.raises(ValueError):
        d.connect('AA', identity, identity.o1, A_pp=identity.i1)

    # Node 'A' exists, but it is not a part of the signature
    # (hidden by previously connected transformation).
    with pytest.raises(ValueError):
        d.connect('B', identity, identity.o1, A=identity.i1)


def test_type_mismatch():
    """Check that the error is thrown if the connection is made to an wrong type."""

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)
    identity = tr_identity(Type(numpy.complex64, (N, N + 1)))

    with pytest.raises(ValueError):
        d.connect('A', identity, identity.o1, A_prime=identity.i1)


def test_wrong_data_path():
    """
    Check that the error is thrown if the connector is a part of the signature,
    but this particular data path (input or output) is already hidden
    by a previously connected transformation.
    """

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = DummyAdvanced(arr_type, coeff_dtype)
    identity = tr_identity(d.parameter.C)

    d.parameter.C.connect(identity, identity.o1, C_in=identity.i1)
    d.parameter.D.connect(identity, identity.i1, D_out=identity.o1)
    assert list(d.signature.parameters.values()) == [
        Parameter('C', Annotation(arr_type, 'o')),
        Parameter('C_in', Annotation(arr_type, 'i')),
        Parameter('D_out', Annotation(arr_type, 'o')),
        Parameter('D', Annotation(arr_type, 'i')),
        Parameter('coeff1', Annotation(coeff_dtype)),
        Parameter('coeff2', Annotation(coeff_dtype))]

    # Now input to C is hidden by the previously connected transformation
    with pytest.raises(ValueError):
        d.parameter.C.connect(identity, identity.o1, C_in_prime=identity.i1)

    # Same goes for D
    with pytest.raises(ValueError):
        d.parameter.D.connect(identity, identity.i1, D_out_prime=identity.o1)

    # Also we cannot make one of the transformation outputs an existing output parameter
    with pytest.raises(ValueError):
        d.parameter.C.connect(identity, identity.i1, D_out=identity.o1)

    # Output of C is still available though
    d.parameter.C.connect(identity, identity.i1, C_out=identity.o1)
    assert list(d.signature.parameters.values()) == [
        Parameter('C_out', Annotation(arr_type, 'o')),
        Parameter('C_in', Annotation(arr_type, 'i')),
        Parameter('D_out', Annotation(arr_type, 'o')),
        Parameter('D', Annotation(arr_type, 'i')),
        Parameter('coeff1', Annotation(coeff_dtype)),
        Parameter('coeff2', Annotation(coeff_dtype))]


def test_wrong_transformation_parameters():
    """
    Check that the error is thrown if the list of transformation parameter names
    does not coincide with actual names.
    """

    N = 200
    coeff_dtype = numpy.float32
    arr_type = Type(numpy.complex64, (N, N))

    d = Dummy(arr_type, arr_type, coeff_dtype, same_A_B=True)
    identity = tr_identity(d.parameter.A)

    # ``identity`` does not have ``input`` parameter
    with pytest.raises(ValueError):
        d.parameter.A.connect(identity, identity.o1, A_prime='input')

    # ``identity`` does not have ``i2`` parameter
    with pytest.raises(ValueError):
        d.parameter.A.connect(identity, identity.o1, A_prime=identity.i1, A2='i2')


def test_io_merge(some_queue):
    """
    Check that one can end input and output transformation in the same node
    thus making its role 'io'.
    """

    N = 200
    coeff1 = 3
    coeff2 = 4
    scale_in = 0.5
    scale_out = 0.3
    arr_type = Type(numpy.complex64, (N, N))
    coeff_dtype = numpy.float32

    C = get_test_array_like(arr_type)
    D = get_test_array_like(arr_type)
    C_dev = Array.from_host(some_queue, C)
    D_dev = Array.from_host(some_queue, D)

    d = DummyAdvanced(C, coeff_dtype)
    scale = tr_scale(d.parameter.C, d.parameter.coeff1.dtype)
    d.parameter.C.connect(scale, scale.o1, C_prime=scale.i1, scale_in=scale.s1)
    d.parameter.C.connect(scale, scale.i1, C_prime=scale.o1, scale_out=scale.s1)
    assert list(d.signature.parameters.values()) == [
        Parameter('C_prime', Annotation(arr_type, 'io')),
        Parameter('scale_out', Annotation(coeff_dtype)),
        Parameter('scale_in', Annotation(coeff_dtype)),
        Parameter('D', Annotation(arr_type, 'io')),
        Parameter('coeff1', Annotation(coeff_dtype)),
        Parameter('coeff2', Annotation(coeff_dtype))]

    dc = d.compile(some_queue.device)
    dc(some_queue, C_dev, scale_out, scale_in, D_dev, coeff1, coeff2)

    C_ref, D_ref = mock_dummy_advanced(C * scale_in, D, coeff1, coeff2)
    C_ref *= scale_out

    C = C_dev.get(some_queue)
    D = D_dev.get(some_queue)

    assert diff_is_negligible(C, C_ref)
    assert diff_is_negligible(D, D_ref)


class ExpressionIndexing(Computation):
    """
    A computation for the test below, with expressions passed to indexing functions.
    Used to check that macro parameters in transformation modules are properly decorated.
    """

    def __init__(self, arr_t):
        assert len(arr_t.shape) == 2

        # reset strides/offset of the input and return a contiguous array
        res_t = Type(arr_t.dtype, arr_t.shape)
        Computation.__init__(self, [
            Parameter('output', Annotation(res_t, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))
            ])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()

        template = template_from("""
        <%def name="kernel(kernel_declaration, output, input)">
        ${kernel_declaration}
        {
            if (${static.skip}()) return;
            VSIZE_T idx0 = ${static.global_id}(0);
            VSIZE_T idx1 = ${static.global_id}(1);

            ${output.ctype} a = ${input.load_idx}(idx0 + 1 - 1, idx1);
            ${output.store_idx}(idx0 + 1 - 1, idx1, a * 2);
        }
        </%def>
        """)

        plan.kernel_call(
            template.get_def('kernel'),
            [output, input_],
            global_size=output.shape)

        return plan


def test_transformation_macros(queue):
    """
    Regression test for #27.
    When expressions are passed to leaf load_idx/store_idx macros,
    they are not processed correctly, because the corresponding parameters are used
    without parenthesis in their bodies.
    Namely, the error happens when the flat index is generated out of per-dimension indices.
    """

    N = 1000
    dtype = numpy.float32

    # The array should be 2D in order for the flat index generation expression to be non-trivial.
    a = get_test_array((N, 2), dtype)

    comp = ExpressionIndexing(a)
    a_dev = Array.from_host(queue, a)
    res_dev = Array.empty_like(queue.device, comp.parameter.output)

    compc = comp.compile(queue.device)
    compc(queue, res_dev, a_dev)

    res_ref = a * 2

    assert diff_is_negligible(res_dev.get(queue), res_ref)


def test_array_views(queue):

    a = get_test_array((6, 8, 10), numpy.int32)

    a_dev = Array.from_host(queue, a)
    b_dev = Array.empty_like(queue.device, a)

    in_view = a_dev[2:4, ::2, ::-1]
    out_view = b_dev[4:, 1:5, :]

    move = PureParallel.from_trf(
        transformations.copy(in_view, out_arr_t=out_view),
        guiding_array='output').compile(queue.device)

    move(queue, out_view, in_view)
    b_res = b_dev.get(queue)[4:, 1:5, :]
    b_ref = a[2:4, ::2, ::-1]

    assert diff_is_negligible(b_res, b_ref)
