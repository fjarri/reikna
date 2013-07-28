import numpy
import pytest

from reikna.core import Parameter, Annotation, Transformation
from reikna.core.signature import Type
from reikna.cluda import functions

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
            cast=functions.cast(arr.dtype, scalar.dtype)))

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


def test_same_shape(thr):

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
    dc = d.compile(thr)

    A_prime = get_test_array_like(d.parameter.A_prime)
    B_new_prime = get_test_array_like(d.parameter.B_new_prime)

    A_prime_dev = thr.to_device(A_prime)
    B_new_prime_dev = thr.to_device(B_new_prime)
    C_new_half1_dev = thr.empty_like(d.parameter.A_prime)
    C_half2_dev = thr.empty_like(d.parameter.A_prime)
    D_prime_dev = thr.empty_like(d.parameter.A_prime)

    dc(
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

    assert diff_is_negligible(C_new_half1_dev.get(), C_new_half1)
    assert diff_is_negligible(C_half2_dev.get(), C_half2)
    assert diff_is_negligible(D_prime_dev.get(), D_prime)


def test_connection_to_base(thr):

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

    dc = d.compile(thr)

    B = get_test_array_like(d.parameter.B)
    B_dev = thr.to_device(B)
    C_prime_dev = thr.empty_like(d.parameter.B)
    D_dev = thr.empty_like(d.parameter.B)

    dc(C_prime_dev, D_dev, B_dev, coeff)

    C, D = mock_dummy(B, B, coeff)
    C_prime = C * coeff

    assert diff_is_negligible(C_prime_dev.get(), C_prime)
    assert diff_is_negligible(D_dev.get(), D)


def test_nested_same_shape(thr):

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
    dc = d.compile(thr)

    A_prime = get_test_array_like(d.parameter.A_prime)
    B_new_prime = get_test_array_like(d.parameter.B_new_prime)

    A_prime_dev = thr.to_device(A_prime)
    B_new_prime_dev = thr.to_device(B_new_prime)
    C_new_half1_dev = thr.empty_like(d.parameter.A_prime)
    C_half2_dev = thr.empty_like(d.parameter.A_prime)
    D_prime_dev = thr.empty_like(d.parameter.A_prime)

    dc(
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

    assert diff_is_negligible(C_new_half1_dev.get(), C_new_half1)
    assert diff_is_negligible(C_half2_dev.get(), C_half2)
    assert diff_is_negligible(D_prime_dev.get(), D_prime)


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


def test_io_merge(some_thr):
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
    C_dev = some_thr.to_device(C)
    D_dev = some_thr.to_device(D)

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

    dc = d.compile(some_thr)
    dc(C_dev, scale_out, scale_in, D_dev, coeff1, coeff2)

    C_ref, D_ref = mock_dummy_advanced(C * scale_in, D, coeff1, coeff2)
    C_ref *= scale_out

    C = C_dev.get()
    D = D_dev.get()

    assert diff_is_negligible(C, C_ref)
    assert diff_is_negligible(D, D_ref)
