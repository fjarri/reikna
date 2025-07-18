"""
An example demonstrating how to build several specialized FFT computations:
- a real-to-complex and complex-to-real FFT using a half-sized complex-to-complex FFT;
- a real-to-complex and complex-to-real FFT with the real signal being anti-periodic
  (x[k] = -x[N/2+k], k=0..N/2-1) using a quarter-sized complex-to-complex FFT.

Note that in the second case all the harmonics with even (0-based) indices are equal to zero,
so both the reference and the GPU FFT return only N//4 odd harmonics
(and, correspondingly, the inverse functions take an N//4-array of odd harmonics
and return the first half of the initial array).

For the sake of the example only even N and 1D FFT are supported.

Source of the algorithms:
L. R. Rabiner, "On the Use of Symmetry in FFT Computation"
IEEE Transactions on Acoustics, Speech, and Signal Processing 27(3), 233-239 (1979)
doi: 10.1109/TASSP.1979.1163235
"""

from collections.abc import Callable

import numpy
from grunnur import (
    API,
    Array,
    ArrayMetadata,
    AsArrayMetadata,
    Context,
    DeviceParameters,
    Queue,
    Template,
    dtypes,
    functions,
)
from numpy.typing import NDArray

from reikna import helpers
from reikna.algorithms import Reduce, Scan, predicate_sum
from reikna.core import (
    Annotation,
    Computation,
    ComputationPlan,
    KernelArgument,
    KernelArguments,
    Parameter,
    Transformation,
    Type,
)
from reikna.fft import FFT

TEMPLATE = Template.from_associated_file(__file__)


# Refernce functions


def rfft_reference(a: NDArray[numpy.floating]) -> NDArray[numpy.complex128]:
    # Numpy already has a dedicated rfft() function, but this one illustrates the algorithm
    # used by the computation.

    assert a.size % 2 == 0  # noqa: S101
    assert a.ndim == 1  # noqa: S101

    size = a.size

    wn_mk = numpy.exp(-2j * numpy.pi * numpy.arange(size // 2) / size)
    a_fft = 0.5 * (1 - 1j * wn_mk)
    b_fft = 0.5 * (1 + 1j * wn_mk)

    x = a[::2] + 1j * a[1::2]
    x_fft = numpy.fft.fft(x)  # size/2-sized FFT

    res = numpy.empty(size // 2 + 1, numpy.complex128)
    res[: size // 2] = x_fft * a_fft + (numpy.roll(x_fft[size // 2 - 1 :: -1], 1)).conj() * b_fft
    res[size // 2] = x_fft[0].real - x_fft[0].imag

    return res


def irfft_reference(a: NDArray[numpy.complexfloating]) -> NDArray[numpy.float64]:
    # Numpy already has a dedicated irfft() function, but this one illustrates the algorithm
    # used by the computation.

    assert a.size % 2 == 1  # noqa: S101
    assert a.ndim == 1  # noqa: S101

    size = (a.size - 1) * 2

    # Following numpy.fft.irftt() which ignores these values
    a[0] = a[0].real
    a[-1] = a[-1].real

    wn_mk = numpy.exp(-2j * numpy.pi * numpy.arange(size // 2) / size)
    a_fft = 0.5 * (1 - 1j * wn_mk)
    b_fft = 0.5 * (1 + 1j * wn_mk)

    a_fft = a_fft.conj()
    b_fft = b_fft.conj()

    x_fft = a[:-1] * a_fft + a[size // 2 : 0 : -1].conj() * b_fft

    x = numpy.fft.ifft(x_fft)  # size/2-sized IFFT

    res = numpy.empty(size, numpy.float64)
    res[::2] = x.real
    res[1::2] = x.imag

    return res


def aprfft_reference(x: NDArray[numpy.floating]) -> NDArray[numpy.complex128]:
    # x : real N/2-array (first half of the full signal)
    # output: complex N/4-array (odd harmonics)

    assert x.size % 2 == 0  # noqa: S101
    assert x.ndim == 1  # noqa: S101

    size = x.size * 2

    y = x * (4 * numpy.sin(2 * numpy.pi * numpy.arange(size // 2) / size))
    y_fft = numpy.fft.rfft(y)

    t = x * numpy.cos(2 * numpy.pi * numpy.arange(size // 2) / size)
    re_x_fft_1 = 2 * sum(t)

    y_fft *= -1j
    y_fft[0] /= 2
    y_fft[0] += re_x_fft_1
    return numpy.cumsum(y_fft[:-1])


def iaprfft_reference(x_fft: NDArray[numpy.complexfloating]) -> NDArray[numpy.float64]:
    # x_fft : complex N/4-array of odd harmonics
    # output: real N/2-array (first half of the signal)

    assert x_fft.ndim == 1  # noqa: S101

    size = x_fft.size * 4
    y_fft = numpy.empty(size // 4 + 1, numpy.complex128)
    y_fft[1:-1] = x_fft[1:] - x_fft[:-1]
    y_fft *= 1j
    y_fft[0] = -2 * x_fft[0].imag
    y_fft[-1] = 2 * x_fft[-1].imag
    y = numpy.fft.irfft(y_fft)

    res = numpy.empty(size // 2, numpy.float64)
    res[1:] = y[1:] / 4 / numpy.sin(2 * numpy.pi * numpy.arange(1, size // 2) / size)
    res[0] = x_fft.real.sum() / (size / 2)

    return res


# GPU computations


def prepare_rfft_input(arr: KernelArgument) -> Transformation:
    res = ArrayMetadata((*arr.shape[:-1], arr.shape[-1] // 2), dtypes.complex_for(arr.dtype))
    return Transformation(
        [
            Parameter("output", Annotation(res, "o")),
            Parameter("input", Annotation(arr, "i")),
        ],
        """
        <%
            batch_idxs = " ".join((idx + ", ") for idx in idxs[:-1])
        %>
        ${input.ctype} re = ${input.load_idx}(${batch_idxs} ${idxs[-1]} * 2);
        ${input.ctype} im = ${input.load_idx}(${batch_idxs} ${idxs[-1]} * 2 + 1);
        ${output.store_same}(COMPLEX_CTR(${output.ctype})(re, im));
        """,
        connectors=["output"],
    )


class RFFT(Computation):
    def __init__(self, arr: AsArrayMetadata, *, dont_store_last: bool = False):
        metadata = arr.as_array_metadata()

        self._dont_store_last = dont_store_last

        output_size = metadata.shape[-1] // 2 + (0 if dont_store_last else 1)

        out_arr = ArrayMetadata(
            (*metadata.shape[:-1], output_size), dtypes.complex_for(metadata.dtype)
        )

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(out_arr, "o")),
                Parameter("input", Annotation(metadata, "i")),
            ],
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        _device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        input_ = args.input
        output = args.output

        size = input_.shape[-1]
        wn_mk = numpy.exp(-2j * numpy.pi * numpy.arange(size // 2) / size)
        a_fft = 0.5 * (1 - 1j * wn_mk)
        b_fft = 0.5 * (1 + 1j * wn_mk)

        a_fft_arr = plan.persistent_array(a_fft.astype(output.dtype))
        b_fft_arr = plan.persistent_array(b_fft.astype(output.dtype))

        cfft_arr = ArrayMetadata((*input_.shape[:-1], input_.shape[-1] // 2), output.dtype)
        cfft = FFT(cfft_arr, axes=(len(input_.shape) - 1,))

        prepare_input = prepare_rfft_input(input_)

        cfft.parameter.input.connect(
            prepare_input, prepare_input.parameter.output, real_input=prepare_input.parameter.input
        )

        temp = plan.temp_array_like(cfft.parameter.output)

        batch_size = helpers.product(output.shape[:-1])

        plan.computation_call(cfft, temp, input_)
        plan.kernel_call(
            TEMPLATE.get_def("prepare_rfft_output"),
            [output, temp, a_fft_arr, b_fft_arr],
            global_size=(batch_size, size // 2),
            render_kwds=dict(
                slices=(len(input_.shape) - 1, 1),
                N=size,
                mul=functions.mul(output.dtype, output.dtype),
                conj=functions.conj(output.dtype),
                dont_store_last=self._dont_store_last,
            ),
        )

        return plan


def prepare_irfft_output(arr: AsArrayMetadata) -> Transformation:
    metadata = arr.as_array_metadata()
    res = ArrayMetadata(
        (*metadata.shape[:-1], metadata.shape[-1] * 2), dtypes.real_for(metadata.dtype)
    )
    return Transformation(
        [
            Parameter("output", Annotation(res, "o")),
            Parameter("input", Annotation(metadata, "i")),
        ],
        """
        <%
            batch_idxs = " ".join((idx + ", ") for idx in idxs[:-1])
        %>
        ${input.ctype} x = ${input.load_same};
        ${output.store_idx}(${batch_idxs} ${idxs[-1]} * 2, x.x);
        ${output.store_idx}(${batch_idxs} ${idxs[-1]} * 2 + 1, x.y);
        """,
        connectors=["output"],
    )


class IRFFT(Computation):
    def __init__(self, arr: AsArrayMetadata):
        metadata = arr.as_array_metadata()
        output_size = (metadata.shape[-1] - 1) * 2
        out_arr = ArrayMetadata(
            (*metadata.shape[:-1], output_size), dtypes.real_for(metadata.dtype)
        )

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(out_arr, "o")),
                Parameter("input", Annotation(metadata, "i")),
            ],
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        _device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        input_ = args.input
        output = args.output

        size = (input_.shape[-1] - 1) * 2

        wn_mk = numpy.exp(-2j * numpy.pi * numpy.arange(size // 2) / size)
        a_fft = 0.5 * (1 - 1j * wn_mk)
        b_fft = 0.5 * (1 + 1j * wn_mk)

        a_fft_arr = plan.persistent_array(a_fft.conj().astype(dtypes.complex_for(output.dtype)))
        b_fft_arr = plan.persistent_array(b_fft.conj().astype(dtypes.complex_for(output.dtype)))

        cfft_arr = ArrayMetadata((*input_.shape[:-1], size // 2), input_.dtype)
        cfft = FFT(cfft_arr, axes=(len(input_.shape) - 1,))

        prepare_output = prepare_irfft_output(cfft.parameter.output)

        cfft.parameter.output.connect(
            prepare_output,
            prepare_output.parameter.input,
            real_output=prepare_output.parameter.output,
        )

        temp = plan.temp_array_like(cfft.parameter.input)

        batch_size = helpers.product(output.shape[:-1])

        plan.kernel_call(
            TEMPLATE.get_def("prepare_irfft_input"),
            [temp, input_, a_fft_arr, b_fft_arr],
            global_size=(batch_size, size // 2),
            render_kwds=dict(
                slices=(len(input_.shape) - 1, 1),
                N=size,
                mul=functions.mul(input_.dtype, input_.dtype),
                conj=functions.conj(input_.dtype),
            ),
        )

        plan.computation_call(cfft, output, temp, inverse=True)

        return plan


def get_multiply(output: AsArrayMetadata) -> Transformation:
    metadata = output.as_array_metadata()
    return Transformation(
        [
            Parameter("output", Annotation(metadata, "o")),
            Parameter("a", Annotation(metadata, "i")),
            Parameter("b", Annotation(ArrayMetadata(metadata.shape[-1], metadata.dtype), "i")),
        ],
        """
        ${output.store_same}(${mul}(${a.load_same}, ${b.load_idx}(${idxs[-1]})));
        """,
        connectors=["output", "a"],
        render_kwds=dict(mul=functions.mul(metadata.dtype, metadata.dtype)),
    )


def get_prepare_prfft_scan(output: AsArrayMetadata) -> Transformation:
    metadata = output.as_array_metadata()
    return Transformation(
        [
            Parameter("output", Annotation(metadata, "o")),
            Parameter("y_fft", Annotation(metadata, "i")),
            Parameter(
                "re_x_fft_0",
                Annotation(
                    ArrayMetadata((*metadata.shape[:-1], 1), dtypes.real_for(metadata.dtype)), "i"
                ),
            ),
        ],
        """
        ${y_fft.ctype} y_fft = ${y_fft.load_same};
        y_fft = COMPLEX_CTR(${y_fft.ctype})(y_fft.y, -y_fft.x);

        if (${idxs[-1]} == 0)
        {
            y_fft.x = y_fft.x / 2 + ${re_x_fft_0.load_idx}(${", ".join(idxs[:-1] + ["0"])});
            y_fft.y /= 2;
        }

        ${output.store_same}(y_fft);
        """,
        connectors=["output", "y_fft"],
    )


class APRFFT(Computation):
    """FFT of a real antiperiodic signal (x[k] = -x[N/2+k])."""

    def __init__(self, arr: AsArrayMetadata):
        metadata = arr.as_array_metadata()
        out_arr = ArrayMetadata(
            (*metadata.shape[:-1], metadata.shape[-1] // 2),
            dtypes.complex_for(metadata.dtype),
        )

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(out_arr, "o")),
                Parameter("input", Annotation(metadata, "i")),
            ],
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        _device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        input_ = args.input
        output = args.output

        size = input_.shape[-1] * 2

        coeffs1 = 4 * numpy.sin(2 * numpy.pi * numpy.arange(size // 2) / size)
        coeffs2 = 2 * numpy.cos(2 * numpy.pi * numpy.arange(size // 2) / size)

        c1_arr = plan.persistent_array(coeffs1.astype(input_.dtype))
        c2_arr = plan.persistent_array(coeffs2.astype(input_.dtype))

        multiply = get_multiply(input_)

        # re_x_fft_1 = sum(x * coeffs2)

        t = plan.temp_array_like(input_)
        rd = Reduce(t, predicate_sum(input_.dtype), axes=(len(input_.shape) - 1,))

        rd.parameter.input.connect(
            multiply, multiply.parameter.output, x=multiply.parameter.a, c2=multiply.parameter.b
        )

        re_x_fft_0 = plan.temp_array_like(rd.parameter.output)
        plan.computation_call(rd, re_x_fft_0, input_, c2_arr)

        # y_fft = numpy.fft.rfft(x * coeffs1)

        rfft = RFFT(input_, dont_store_last=True)
        rfft.parameter.input.connect(
            multiply, multiply.parameter.output, x=multiply.parameter.a, c1=multiply.parameter.b
        )

        y_fft = plan.temp_array_like(rfft.parameter.output)
        plan.computation_call(rfft, y_fft, input_, c1_arr)

        # y_fft *= -1j
        # y_fft[0] /= 2
        # y_fft[0] += re_x_fft_1
        # res = numpy.cumsum(y_fft[:-1])

        prepare_prfft_scan = get_prepare_prfft_scan(y_fft)

        sc = Scan(y_fft, predicate_sum(y_fft.dtype), axes=(-1,), exclusive=False)
        sc.parameter.input.connect(
            prepare_prfft_scan,
            prepare_prfft_scan.parameter.output,
            y_fft=prepare_prfft_scan.parameter.y_fft,
            re_x_fft_0=prepare_prfft_scan.parameter.re_x_fft_0,
        )

        plan.computation_call(sc, output, y_fft, re_x_fft_0)

        return plan


def get_prepare_iprfft_input(input_: AsArrayMetadata) -> Transformation:
    # Input: size N//4
    # Output: size N//4+1

    metadata = input_.as_array_metadata()

    size = metadata.shape[-1] * 4
    output = ArrayMetadata((*metadata.shape[:-1], size // 4 + 1), metadata.dtype)

    return Transformation(
        [
            Parameter("y_fft", Annotation(output, "o")),
            Parameter("x_fft", Annotation(metadata, "i")),
        ],
        """
        <%
            batch_idxs = " ".join((idx + ", ") for idx in idxs[:-1])
        %>

        ${y_fft.ctype} y_fft;
        if (${idxs[-1]} == 0)
        {
            ${x_fft.ctype} x_fft = ${x_fft.load_idx}(${batch_idxs} 0);
            y_fft = COMPLEX_CTR(${y_fft.ctype})(-2 * x_fft.y, 0);
        }
        else if (${idxs[-1]} == ${N//4})
        {
            ${x_fft.ctype} x_fft = ${x_fft.load_idx}(${batch_idxs} ${N//4-1});
            y_fft = COMPLEX_CTR(${y_fft.ctype})(2 * x_fft.y, 0);
        }
        else
        {
            ${x_fft.ctype} x_fft = ${x_fft.load_idx}(${batch_idxs} ${idxs[-1]});
            ${x_fft.ctype} x_fft_prev = ${x_fft.load_idx}(${batch_idxs} ${idxs[-1]} - 1);
            ${x_fft.ctype} diff = x_fft - x_fft_prev;
            y_fft = COMPLEX_CTR(${y_fft.ctype})(-diff.y, diff.x);
        }

        ${y_fft.store_same}(y_fft);
        """,
        connectors=["y_fft"],
        render_kwds=dict(N=size),
    )


def get_prepare_iprfft_output(y: AsArrayMetadata) -> Transformation:
    # Input: size N//4
    # Output: size N//4

    metadata = y.as_array_metadata()

    size = metadata.shape[-1] * 2

    return Transformation(
        [
            Parameter("x", Annotation(metadata, "o")),
            Parameter("y", Annotation(metadata, "i")),
            Parameter(
                "x0", Annotation(ArrayMetadata((*metadata.shape[:-1], 1), metadata.dtype), "i")
            ),
            Parameter("coeffs", Annotation(ArrayMetadata(size // 2, metadata.dtype), "i")),
        ],
        """
        ${y.ctype} y = ${y.load_same};
        ${coeffs.ctype} coeff = ${coeffs.load_idx}(${idxs[-1]});

        ${x.ctype} x;

        if (${idxs[-1]} == 0)
        {
            ${x0.ctype} x0 = ${x0.load_idx}(${", ".join(idxs[:-1] + ["0"])});
            x = x0 / ${N // 2};
        }
        else
        {
            x = y * coeff;
        }

        ${x.store_same}(x);
        """,
        connectors=["y"],
        render_kwds=dict(N=size),
    )


class IAPRFFT(Computation):
    """IFFT of a real antiperiodic signal (x[k] = -x[N/2+k])."""

    def __init__(self, arr: AsArrayMetadata):
        metadata = arr.as_array_metadata()
        out_arr = ArrayMetadata(
            (*metadata.shape[:-1], metadata.shape[-1] * 2), dtypes.real_for(metadata.dtype)
        )

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(out_arr, "o")),
                Parameter("input", Annotation(metadata, "i")),
            ],
        )

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        _device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        input_ = args.input
        output = args.output

        size = input_.shape[-1] * 4

        # The first element is unused
        coeffs = numpy.concatenate(
            [[0], 1 / (4 * numpy.sin(2 * numpy.pi * numpy.arange(1, size // 2) / size))]
        ).astype(numpy.float32)
        coeffs_arr = plan.persistent_array(coeffs)

        prepare_iprfft_input = get_prepare_iprfft_input(input_)
        prepare_iprfft_output = get_prepare_iprfft_output(output)

        irfft = IRFFT(prepare_iprfft_input.parameter.y_fft)
        irfft.parameter.input.connect(
            prepare_iprfft_input,
            prepare_iprfft_input.parameter.y_fft,
            x_fft=prepare_iprfft_input.parameter.x_fft,
        )
        irfft.parameter.output.connect(
            prepare_iprfft_output,
            prepare_iprfft_output.parameter.y,
            x=prepare_iprfft_output.parameter.x,
            x0=prepare_iprfft_output.parameter.x0,
            coeffs=prepare_iprfft_output.parameter.coeffs,
        )

        real = Transformation(
            [
                Parameter(
                    "output",
                    Annotation(ArrayMetadata(input_.shape, dtypes.real_for(input_.dtype)), "o"),
                ),
                Parameter("input", Annotation(input_, "i")),
            ],
            """
            ${output.store_same}((${input.load_same}).x);
            """,
            connectors=["output"],
        )

        rd_t = ArrayMetadata(input_.shape, output.dtype)
        rd = Reduce(rd_t, predicate_sum(rd_t.dtype), axes=(len(input_.shape) - 1,))
        rd.parameter.input.connect(real, real.parameter.output, x_fft=real.parameter.input)

        x0 = plan.temp_array_like(rd.parameter.output)

        plan.computation_call(rd, x0, input_)
        plan.computation_call(irfft, output, x0, coeffs_arr, input_)

        return plan


# Tests


def test_rfft(queue: Queue) -> None:
    size = 1024
    rng = numpy.random.default_rng()
    a = rng.normal(size=size).astype(numpy.float32)
    a_dev = Array.from_host(queue, a)

    rfft = RFFT(a_dev).compile(queue.device)

    fa_numpy = numpy.fft.rfft(a)
    fa_ref = rfft_reference(a)

    fa_gpu = Array.empty_like(queue.device, rfft.parameter.output)
    rfft(queue, fa_gpu, a_dev)

    assert numpy.allclose(fa_numpy, fa_ref)  # noqa: S101
    assert numpy.allclose(fa_numpy, fa_gpu.get(queue), atol=1e-4, rtol=1e-4)  # noqa: S101


def test_irfft(queue: Queue) -> None:
    size = 1024
    rng = numpy.random.default_rng()
    fa = (rng.normal(size=size // 2 + 1) + 1j * rng.normal(size=size // 2 + 1)).astype(
        numpy.complex64
    )
    fa_dev = Array.from_host(queue, fa)

    irfft = IRFFT(fa_dev).compile(queue.device)

    a_numpy = numpy.fft.irfft(fa)
    a_ref = irfft_reference(fa)

    a_gpu = Array.empty_like(queue.device, irfft.parameter.output)
    irfft(queue, a_gpu, fa_dev)

    assert numpy.allclose(a_numpy, a_ref)  # noqa: S101
    assert numpy.allclose(a_numpy, a_gpu.get(queue), atol=1e-4, rtol=1e-4)  # noqa: S101


def test_aprfft(queue: Queue) -> None:
    size = 1024
    rng = numpy.random.default_rng()
    half_a = rng.normal(size=size // 2).astype(numpy.float32)
    a = numpy.concatenate([half_a, -half_a])

    half_a_dev = Array.from_host(queue, half_a)

    aprfft = APRFFT(half_a_dev).compile(queue.device)

    fa_numpy = numpy.fft.rfft(a)[1::2]
    fa_ref = aprfft_reference(half_a)

    fa_dev = Array.empty_like(queue.device, aprfft.parameter.output)
    aprfft(queue, fa_dev, half_a_dev)

    assert numpy.allclose(fa_numpy, fa_ref)  # noqa: S101
    assert numpy.allclose(fa_numpy, fa_dev.get(queue), atol=1e-4, rtol=1e-4)  # noqa: S101


def test_iaprfft(queue: Queue) -> None:
    size = 1024
    rng = numpy.random.default_rng()
    fa_odd_harmonics = (rng.normal(size=size // 4) + 1j * rng.normal(size=size // 4)).astype(
        numpy.complex64
    )

    fa = numpy.zeros(size // 2 + 1, fa_odd_harmonics.dtype)
    fa[1::2] = fa_odd_harmonics

    fa_oh_dev = Array.from_host(queue, fa_odd_harmonics)

    iaprfft = IAPRFFT(fa_oh_dev).compile(queue.device)

    half_a_numpy = numpy.fft.irfft(fa)[: size // 2]
    half_a_ref = iaprfft_reference(fa_odd_harmonics)

    half_a_dev = Array.empty_like(queue.device, iaprfft.parameter.output)
    iaprfft(queue, half_a_dev, fa_oh_dev)

    assert numpy.allclose(half_a_numpy, half_a_ref, atol=1e-4, rtol=1e-4)  # noqa: S101
    assert numpy.allclose(half_a_numpy, half_a_dev.get(queue), atol=1e-4, rtol=1e-4)  # noqa: S101


if __name__ == "__main__":
    context = Context.from_devices([API.any().platforms[0].devices[0]])
    queue = Queue(context.device)

    test_rfft(queue)
    test_irfft(queue)
    test_aprfft(queue)
    test_iaprfft(queue)
