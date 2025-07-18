import re

import numpy
import pytest
from grunnur import Array, ArrayMetadata, Template, functions

from helpers import diff_is_negligible, get_test_array, get_test_array_like
from reikna.algorithms import PureParallel
from reikna.core import Annotation, Computation, Parameter, Transformation
from reikna.core.signature import Type
from test_core.dummy import (
    Dummy,
    DummyAdvanced,
    DummyNested,
    mock_dummy,
    mock_dummy_advanced,
    mock_dummy_nested,
    tr_scale,
)

# Some transformations to use by tests


# Identity transformation: Output = Input
def tr_identity(arr):
    return Transformation(
        [Parameter("o1", Annotation(arr, "o")), Parameter("i1", Annotation(arr, "i"))],
        "${o1.store_same}(${i1.load_same});",
    )


# Output = Input1 * Parameter1 + Input 2
def tr_2_to_1(arr, scalar):
    return Transformation(
        [
            Parameter("o1", Annotation(arr, "o")),
            Parameter("i1", Annotation(arr, "i")),
            Parameter("i2", Annotation(arr, "i")),
            Parameter("s1", Annotation(scalar)),
        ],
        """
        ${o1.ctype} t = ${mul}(${cast}(${s1}), ${i1.load_same});
        ${o1.store_same}(t + ${i2.load_same});
        """,
        render_kwds=dict(
            mul=functions.mul(arr.dtype, arr.dtype), cast=functions.cast(scalar.dtype, arr.dtype)
        ),
    )


# Output1 = Input / 2, Output2 = Input / 2
def tr_1_to_2(arr):
    return Transformation(
        [
            Parameter("o1", Annotation(arr, "o")),
            Parameter("o2", Annotation(arr, "o")),
            Parameter("i1", Annotation(arr, "i")),
        ],
        """
        ${o1.ctype} t = ${mul}(${i1.load_same}, 0.5);
        ${o1.store_same}(t);
        ${o2.store_same}(t);
        """,
        render_kwds=dict(mul=functions.mul(arr.dtype, numpy.float32)),
    )


def test_io_parameter_in_transformation():
    with pytest.raises(
        ValueError, match=re.escape("Transformation cannot have 'io' parameters ('o1')")
    ):
        Transformation(
            [Parameter("o1", Annotation(ArrayMetadata(100, numpy.float32), "io"))],
            "${o1.store_same}(${o1.load_same});",
        )


def test_signature_correctness():
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)

    # Root signature
    assert list(dummy.signature.parameters.values()) == [
        Parameter("C", Annotation(arr_type, "o")),
        Parameter("D", Annotation(arr_type, "o")),
        Parameter("A", Annotation(arr_type, "i")),
        Parameter("B", Annotation(arr_type, "i")),
        Parameter("coeff", Annotation(coeff_dtype)),
    ]

    identity = tr_identity(dummy.parameter.A)
    join = tr_2_to_1(dummy.parameter.A, dummy.parameter.coeff)
    split = tr_1_to_2(dummy.parameter.A)
    scale = tr_scale(dummy.parameter.A, dummy.parameter.coeff.dtype)

    # Connect some transformations
    dummy.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)
    dummy.parameter.B.connect(join, join.o1, A_prime=join.i1, B_prime=join.i2, B_param=join.s1)
    dummy.parameter.B_prime.connect(identity, identity.o1, B_new_prime=identity.i1)
    dummy.parameter.C.connect(split, split.i1, C_half1=split.o1, C_half2=split.o2)
    dummy.parameter.C_half1.connect(identity, identity.i1, C_new_half1=identity.o1)
    dummy.parameter.D.connect(scale, scale.i1, D_prime=scale.o1, D_param=scale.s1)

    assert list(dummy.signature.parameters.values()) == [
        Parameter("C_new_half1", Annotation(arr_type, "o")),
        Parameter("C_half2", Annotation(arr_type, "o")),
        Parameter("D_prime", Annotation(arr_type, "o")),
        Parameter("D_param", Annotation(coeff_dtype)),
        Parameter("A_prime", Annotation(arr_type, "i")),
        Parameter("B_new_prime", Annotation(arr_type, "i")),
        Parameter("B_param", Annotation(coeff_dtype)),
        Parameter("coeff", Annotation(coeff_dtype)),
    ]


def test_same_shape(queue):
    size = 200
    coeff = 2
    b_param = 3
    d_param = 4

    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)

    identity = tr_identity(dummy.parameter.A)
    join = tr_2_to_1(dummy.parameter.A, dummy.parameter.coeff)
    split = tr_1_to_2(dummy.parameter.A)
    scale = tr_scale(dummy.parameter.A, dummy.parameter.coeff.dtype)

    dummy.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)
    dummy.parameter.B.connect(join, join.o1, A_prime=join.i1, B_prime=join.i2, B_param=join.s1)
    dummy.parameter.B_prime.connect(identity, identity.o1, B_new_prime=identity.i1)
    dummy.parameter.C.connect(split, split.i1, C_half1=split.o1, C_half2=split.o2)
    dummy.parameter.C_half1.connect(identity, identity.i1, C_new_half1=identity.o1)
    dummy.parameter.D.connect(scale, scale.i1, D_prime=scale.o1, D_param=scale.s1)
    dc = dummy.compile(queue.device)

    a_prime = get_test_array_like(dummy.parameter.A_prime)
    b_new_prime = get_test_array_like(dummy.parameter.B_new_prime)

    a_prime_dev = Array.from_host(queue, a_prime)
    b_new_prime_dev = Array.from_host(queue, b_new_prime)
    c_new_half1_dev = Array.empty_like(queue.device, dummy.parameter.A_prime)
    c_half2_dev = Array.empty_like(queue.device, dummy.parameter.A_prime)
    d_prime_dev = Array.empty_like(queue.device, dummy.parameter.A_prime)

    dc(
        queue,
        c_new_half1_dev,
        c_half2_dev,
        d_prime_dev,
        d_param,
        a_prime_dev,
        b_new_prime_dev,
        b_param,
        coeff,
    )

    a = a_prime
    b = a_prime * b_param + b_new_prime
    c, d = mock_dummy(a, b, coeff)
    c_new_half1 = c / 2
    c_half2 = c / 2
    d_prime = d * d_param

    assert diff_is_negligible(c_new_half1_dev.get(queue), c_new_half1)
    assert diff_is_negligible(c_half2_dev.get(queue), c_half2)
    assert diff_is_negligible(d_prime_dev.get(queue), d_prime)


def test_connection_to_base(queue):
    size = 200
    coeff = 2

    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)

    identity = tr_identity(dummy.parameter.A)
    scale = tr_scale(dummy.parameter.A, dummy.parameter.coeff.dtype)

    # connect to the base array argument (effectively making B the same as A)
    dummy.parameter.A.connect(identity, identity.o1, B=identity.i1)
    # connect to the base scalar argument
    dummy.parameter.C.connect(scale, scale.i1, C_prime=scale.o1, coeff=scale.s1)

    assert list(dummy.signature.parameters.values()) == [
        Parameter("C_prime", Annotation(arr_type, "o")),
        Parameter("D", Annotation(arr_type, "o")),
        Parameter("B", Annotation(arr_type, "i")),
        Parameter("coeff", Annotation(coeff_dtype)),
    ]

    dc = dummy.compile(queue.device)

    b = get_test_array_like(dummy.parameter.B)
    b_dev = Array.from_host(queue, b)
    c_prime_dev = Array.empty_like(queue.device, dummy.parameter.B)
    d_dev = Array.empty_like(queue.device, dummy.parameter.B)

    dc(queue, c_prime_dev, d_dev, b_dev, coeff)

    c, d = mock_dummy(b, b, coeff)
    c_prime = c * coeff

    assert diff_is_negligible(c_prime_dev.get(queue), c_prime)
    assert diff_is_negligible(d_dev.get(queue), d)


def test_nested_same_shape(queue):
    size = 2000
    coeff = 2
    second_coeff = 7
    b_param = 3
    d_param = 4

    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = DummyNested(arr_type, arr_type, coeff_dtype, second_coeff, same_a_b=True)

    identity = tr_identity(dummy.parameter.A)
    join = tr_2_to_1(dummy.parameter.A, dummy.parameter.coeff)
    split = tr_1_to_2(dummy.parameter.A)
    scale = tr_scale(dummy.parameter.A, dummy.parameter.coeff.dtype)

    dummy.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)
    dummy.parameter.B.connect(join, join.o1, A_prime=join.i1, B_prime=join.i2, B_param=join.s1)
    dummy.parameter.B_prime.connect(identity, identity.o1, B_new_prime=identity.i1)
    dummy.parameter.C.connect(split, split.i1, C_half1=split.o1, C_half2=split.o2)
    dummy.parameter.C_half1.connect(identity, identity.i1, C_new_half1=identity.o1)
    dummy.parameter.D.connect(scale, scale.i1, D_prime=scale.o1, d_param=scale.s1)
    dc = dummy.compile(queue.device)

    a_prime = get_test_array_like(dummy.parameter.A_prime)
    b_new_prime = get_test_array_like(dummy.parameter.B_new_prime)

    a_prime_dev = Array.from_host(queue, a_prime)
    b_new_prime_dev = Array.from_host(queue, b_new_prime)
    c_new_half1_dev = Array.empty_like(queue.device, dummy.parameter.A_prime)
    c_half2_dev = Array.empty_like(queue.device, dummy.parameter.A_prime)
    d_prime_dev = Array.empty_like(queue.device, dummy.parameter.A_prime)

    dc(
        queue,
        c_new_half1_dev,
        c_half2_dev,
        d_prime_dev,
        d_param,
        a_prime_dev,
        b_new_prime_dev,
        b_param,
        coeff,
    )

    a = a_prime
    b = a_prime * b_param + b_new_prime
    c, d = mock_dummy_nested(a, b, coeff, second_coeff)
    c_new_half1 = c / 2
    c_half2 = c / 2
    d_prime = d * d_param

    assert diff_is_negligible(c_new_half1_dev.get(queue), c_new_half1)
    assert diff_is_negligible(c_half2_dev.get(queue), c_half2)
    assert diff_is_negligible(d_prime_dev.get(queue), d_prime)


def test_strings_as_parameters():
    """
    Check that one can connect transformations using strings as identifiers
    for computation and transformation parameters.
    """
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)
    identity = tr_identity(dummy.parameter.A)

    dummy.connect("A", identity, "o1", A_prime="i1")

    assert list(dummy.signature.parameters.values()) == [
        Parameter("C", Annotation(arr_type, "o")),
        Parameter("D", Annotation(arr_type, "o")),
        Parameter("A_prime", Annotation(arr_type, "i")),
        Parameter("B", Annotation(arr_type, "i")),
        Parameter("coeff", Annotation(coeff_dtype)),
    ]


def test_alien_parameters():
    """
    Check that one cannot connect transformations using parameter objects from
    other transformation/computation.
    """
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)
    dummy2 = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)
    identity = tr_identity(dummy.parameter.A)
    identity2 = tr_identity(dummy.parameter.A)

    with pytest.raises(
        ValueError, match=re.escape("The connection target must belong to this computation.")
    ):
        dummy.connect(dummy2.parameter.A, identity, "o1", A_prime="i1")

    with pytest.raises(
        ValueError, match="The transformation parameter must belong to the provided transformation"
    ):
        dummy.connect(dummy.parameter.A, identity, identity2.o1, A_prime="i1")

    with pytest.raises(
        ValueError, match="The transformation parameter must belong to the provided transformation"
    ):
        dummy.parameter.A.connect(identity, identity.o1, A_prime=identity2.i1)


def test_connector_repetition():
    """Check that the connector id cannot be repeated in the connections list."""
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)
    identity = tr_identity(dummy.parameter.A)

    with pytest.raises(
        ValueError,
        match=(
            "Parameter 'A' cannot be supplied both as the main connector "
            "and one of the child connections"
        ),
    ):
        dummy.parameter.A.connect(identity, identity.o1, A=identity.o1, A_prime=identity.i1)


def test_wrong_connector():
    """Check that the error is thrown if the connector is unknown or is not in the signature."""
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)
    identity = tr_identity(dummy.parameter.A)

    dummy.parameter.A.connect(identity, identity.o1, A_prime=identity.i1)

    # Connector is missing
    with pytest.raises(ValueError, match="Parameter 'AA' is not a part of the signature"):
        dummy.connect("AA", identity, identity.o1, A_pp=identity.i1)

    # Node 'A' exists, but it is not a part of the signature
    # (hidden by previously connected transformation).
    with pytest.raises(ValueError, match="Parameter 'A' is hidden by transformations"):
        dummy.connect("B", identity, identity.o1, A=identity.i1)


def test_type_mismatch():
    """Check that the error is thrown if the connection is made to an wrong type."""
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)
    identity = tr_identity(ArrayMetadata((size, size + 1), numpy.complex64))

    with pytest.raises(
        ValueError,
        match=r"Incompatible types of the transformation parameter 'o1' .+ and the node 'A'",
    ):
        dummy.connect("A", identity, identity.o1, A_prime=identity.i1)


def test_wrong_data_path():
    """
    Check that the error is thrown if the connector is a part of the signature,
    but this particular data path (input or output) is already hidden
    by a previously connected transformation.
    """
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = DummyAdvanced(arr_type, coeff_dtype)
    identity = tr_identity(dummy.parameter.C)

    dummy.parameter.C.connect(identity, identity.o1, C_in=identity.i1)
    dummy.parameter.D.connect(identity, identity.i1, D_out=identity.o1)
    assert list(dummy.signature.parameters.values()) == [
        Parameter("C", Annotation(arr_type, "o")),
        Parameter("C_in", Annotation(arr_type, "i")),
        Parameter("D_out", Annotation(arr_type, "o")),
        Parameter("D", Annotation(arr_type, "i")),
        Parameter("coeff1", Annotation(coeff_dtype)),
        Parameter("coeff2", Annotation(coeff_dtype)),
    ]

    # Now input to C is hidden by the previously connected transformation
    with pytest.raises(ValueError, match="'C' is not an input node"):
        dummy.parameter.C.connect(identity, identity.o1, C_in_prime=identity.i1)

    # Same goes for D
    with pytest.raises(ValueError, match="'D' is not an output node"):
        dummy.parameter.D.connect(identity, identity.i1, D_out_prime=identity.o1)

    # Also we cannot make one of the transformation outputs an existing output parameter
    with pytest.raises(
        ValueError,
        match="Cannot connect transformation parameter 'o1' to an existing output node 'D_out'",
    ):
        dummy.parameter.C.connect(identity, identity.i1, D_out=identity.o1)

    # Output of C is still available though
    dummy.parameter.C.connect(identity, identity.i1, C_out=identity.o1)
    assert list(dummy.signature.parameters.values()) == [
        Parameter("C_out", Annotation(arr_type, "o")),
        Parameter("C_in", Annotation(arr_type, "i")),
        Parameter("D_out", Annotation(arr_type, "o")),
        Parameter("D", Annotation(arr_type, "i")),
        Parameter("coeff1", Annotation(coeff_dtype)),
        Parameter("coeff2", Annotation(coeff_dtype)),
    ]


def test_wrong_transformation_parameters():
    """
    Check that the error is thrown if the list of transformation parameter names
    does not coincide with actual names.
    """
    size = 200
    coeff_dtype = numpy.float32
    arr_type = ArrayMetadata((size, size), numpy.complex64)

    dummy = Dummy(arr_type, arr_type, coeff_dtype, same_a_b=True)
    identity = tr_identity(dummy.parameter.A)

    # ``identity`` does not have ``input`` parameter
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Supplied transformation names ['input', 'o1'] "
            "do not fully coincide with the existing ones: ['i1', 'o1']"
        ),
    ):
        dummy.parameter.A.connect(identity, identity.o1, A_prime="input")

    # ``identity`` does not have ``i2`` parameter
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Supplied transformation names ['i1', 'i2', 'o1'] "
            "do not fully coincide with the existing ones: ['i1', 'o1']"
        ),
    ):
        dummy.parameter.A.connect(identity, identity.o1, A_prime=identity.i1, A2="i2")


def test_io_merge(some_queue):
    """
    Check that one can end input and output transformation in the same node
    thus making its role 'io'.
    """
    size = 200
    coeff1 = 3
    coeff2 = 4
    scale_in = 0.5
    scale_out = 0.3
    arr_type = ArrayMetadata((size, size), numpy.complex64)
    coeff_dtype = numpy.float32

    c = get_test_array_like(arr_type)
    d = get_test_array_like(arr_type)
    c_dev = Array.from_host(some_queue, c)
    d_dev = Array.from_host(some_queue, d)

    dummy = DummyAdvanced(c_dev, coeff_dtype)
    scale = tr_scale(dummy.parameter.C, dummy.parameter.coeff1.dtype)
    dummy.parameter.C.connect(scale, scale.o1, C_prime=scale.i1, scale_in=scale.s1)
    dummy.parameter.C.connect(scale, scale.i1, C_prime=scale.o1, scale_out=scale.s1)
    assert list(dummy.signature.parameters.values()) == [
        Parameter("C_prime", Annotation(arr_type, "io")),
        Parameter("scale_out", Annotation(coeff_dtype)),
        Parameter("scale_in", Annotation(coeff_dtype)),
        Parameter("D", Annotation(arr_type, "io")),
        Parameter("coeff1", Annotation(coeff_dtype)),
        Parameter("coeff2", Annotation(coeff_dtype)),
    ]

    dc = dummy.compile(some_queue.device)
    dc(some_queue, c_dev, scale_out, scale_in, d_dev, coeff1, coeff2)

    c_ref, d_ref = mock_dummy_advanced(c * scale_in, d, coeff1, coeff2)
    c_ref *= scale_out

    c = c_dev.get(some_queue)
    d = d_dev.get(some_queue)

    assert diff_is_negligible(c, c_ref)
    assert diff_is_negligible(d, d_ref)


class ExpressionIndexing(Computation):
    """
    A computation for the test below, with expressions passed to indexing functions.
    Used to check that macro parameters in transformation modules are properly decorated.
    """

    def __init__(self, arr_t):
        assert len(arr_t.shape) == 2

        # reset strides/offset of the input and return a contiguous array
        res_t = ArrayMetadata(arr_t.shape, arr_t.dtype)
        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(res_t, "o")),
                Parameter("input", Annotation(arr_t, "i")),
            ],
        )

    def _build_plan(self, plan_factory, _device_params, args):
        plan = plan_factory()

        output = args.output
        input_ = args.input

        template = Template.from_string("""
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

        plan.kernel_call(template.get_def("kernel"), [output, input_], global_size=output.shape)

        return plan


def test_transformation_macros(queue):
    """
    Regression test for #27.
    When expressions are passed to leaf load_idx/store_idx macros,
    they are not processed correctly, because the corresponding parameters are used
    without parenthesis in their bodies.
    Namely, the error happens when the flat index is generated out of per-dimension indices.
    """
    size = 1000
    dtype = numpy.float32

    # The array should be 2D in order for the flat index generation expression to be non-trivial.
    a = get_test_array((size, 2), dtype)
    a_dev = Array.from_host(queue, a)

    comp = ExpressionIndexing(a_dev)
    res_dev = Array.empty_like(queue.device, comp.parameter.output)

    compc = comp.compile(queue.device)
    compc(queue, res_dev, a_dev)

    res_ref = a * 2

    assert diff_is_negligible(res_dev.get(queue), res_ref)
