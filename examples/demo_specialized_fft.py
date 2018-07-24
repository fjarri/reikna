"""
This example demonstrates how to build several specialized FFT computations:
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

import numpy

from reikna.fft import FFT

import reikna.helpers as helpers
from reikna.cluda import dtypes, any_api
from reikna.core import Computation, Parameter, Annotation, Type, Transformation
import reikna.cluda.functions as functions
from reikna.algorithms import Reduce, Scan, predicate_sum

TEMPLATE = helpers.template_for(__file__)


# Refernce functions


def rfft_reference(a):
    # Numpy already has a dedicated rfft() function, but this one illustrates the algorithm
    # used by the computation.

    assert a.size % 2 == 0
    assert a.ndim == 1

    N = a.size

    WNmk = numpy.exp(-2j * numpy.pi * numpy.arange(N//2) / N)
    A = 0.5 * (1 - 1j * WNmk)
    B = 0.5 * (1 + 1j * WNmk)

    x = a[::2] + 1j * a[1::2]
    X = numpy.fft.fft(x) # N/2-sized FFT

    res = numpy.empty(N//2 + 1, numpy.complex128)
    res[:N//2] = X * A + (numpy.roll(X[N//2-1::-1], 1)).conj() * B
    res[N//2] = X[0].real - X[0].imag

    return res


def irfft_reference(a):
    # Numpy already has a dedicated irfft() function, but this one illustrates the algorithm
    # used by the computation.

    assert a.size % 2 == 1
    assert a.ndim == 1

    N = (a.size - 1) * 2

    # Following numpy.fft.irftt() which ignores these values
    a[0] = a[0].real
    a[-1] = a[-1].real

    WNmk = numpy.exp(-2j * numpy.pi * numpy.arange(N//2) / N)
    A = 0.5 * (1 - 1j * WNmk)
    B = 0.5 * (1 + 1j * WNmk)

    A = A.conj()
    B = B.conj()

    X = a[:-1] * A + a[N//2:0:-1].conj() * B

    x = numpy.fft.ifft(X) # N/2-sized IFFT

    res = numpy.empty(N, numpy.float64)
    res[::2] = x.real
    res[1::2] = x.imag

    return res


def aprfft_reference(x):
    # x : real N/2-array (first half of the full signal)
    # output: complex N/4-array (odd harmonics)

    assert x.size % 2 == 0
    assert x.ndim == 1

    N = x.size * 2

    y = x * (4 * numpy.sin(2 * numpy.pi * numpy.arange(N//2) / N))
    Y = numpy.fft.rfft(y)

    t = x * numpy.cos(2 * numpy.pi * numpy.arange(N//2) / N)
    re_X_1 = 2 * sum(t)

    Y *= -1j
    Y[0] /= 2
    Y[0] += re_X_1
    res = numpy.cumsum(Y[:-1])

    return res


def iaprfft_reference(X):
    # X : complex N/4-array of odd harmonics
    # output: real N/2-array (first half of the signal)

    assert X.ndim == 1

    N = X.size * 4
    Y = numpy.empty(N//4+1, numpy.complex128)
    Y[1:-1] = X[1:] - X[:-1]
    Y *= 1j
    Y[0] = -2 * X[0].imag
    Y[-1] = 2 * X[-1].imag
    y = numpy.fft.irfft(Y)

    res = numpy.empty(N//2, numpy.float64)
    res[1:] = y[1:] / 4 / numpy.sin(2 * numpy.pi * numpy.arange(1, N//2) / N)
    res[0] = X.real.sum() / (N / 2)

    return res


# GPU computations


def prepare_rfft_input(arr):
    res = Type(dtypes.complex_for(arr.dtype), arr.shape[:-1] + (arr.shape[-1] // 2,))
    return Transformation(
        [
            Parameter('output', Annotation(res, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        <%
            batch_idxs = " ".join((idx + ", ") for idx in idxs[:-1])
        %>
        ${input.ctype} re = ${input.load_idx}(${batch_idxs} ${idxs[-1]} * 2);
        ${input.ctype} im = ${input.load_idx}(${batch_idxs} ${idxs[-1]} * 2 + 1);
        ${output.store_same}(COMPLEX_CTR(${output.ctype})(re, im));
        """,
        connectors=['output'])


class RFFT(Computation):

    def __init__(self, arr_t, dont_store_last=False):
        self._dont_store_last = dont_store_last

        output_size = arr_t.shape[-1] // 2 + (0 if dont_store_last else 1)

        out_arr = Type(
            dtypes.complex_for(arr_t.dtype),
            arr_t.shape[:-1] + (output_size,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = input_.shape[-1]
        WNmk = numpy.exp(-2j * numpy.pi * numpy.arange(N//2) / N)
        A = 0.5 * (1 - 1j * WNmk)
        B = 0.5 * (1 + 1j * WNmk)

        A_arr = plan.persistent_array(A)
        B_arr = plan.persistent_array(B)

        cfft_arr = Type(output.dtype, input_.shape[:-1] + (input_.shape[-1] // 2,))
        cfft = FFT(cfft_arr, axes=(len(input_.shape) - 1,))

        prepare_input = prepare_rfft_input(input_)

        cfft.parameter.input.connect(
            prepare_input, prepare_input.output, real_input=prepare_input.input)

        temp = plan.temp_array_like(cfft.parameter.output)

        batch_size = helpers.product(output.shape[:-1])

        plan.computation_call(cfft, temp, input_)
        plan.kernel_call(
            TEMPLATE.get_def('prepare_rfft_output'),
                [output, temp, A_arr, B_arr],
                global_size=(batch_size, N // 2),
                render_kwds=dict(
                    slices=(len(input_.shape) - 1, 1),
                    N=N,
                    mul=functions.mul(output.dtype, output.dtype),
                    conj=functions.conj(output.dtype),
                    dont_store_last=self._dont_store_last))

        return plan


def prepare_irfft_output(arr):
    res = Type(dtypes.real_for(arr.dtype), arr.shape[:-1] + (arr.shape[-1] * 2,))
    return Transformation(
        [
            Parameter('output', Annotation(res, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        <%
            batch_idxs = " ".join((idx + ", ") for idx in idxs[:-1])
        %>
        ${input.ctype} x = ${input.load_same};
        ${output.store_idx}(${batch_idxs} ${idxs[-1]} * 2, x.x);
        ${output.store_idx}(${batch_idxs} ${idxs[-1]} * 2 + 1, x.y);
        """,
        connectors=['output'])


class IRFFT(Computation):

    def __init__(self, arr_t):

        output_size = (arr_t.shape[-1] - 1) * 2

        out_arr = Type(
            dtypes.real_for(arr_t.dtype),
            arr_t.shape[:-1] + (output_size,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = (input_.shape[-1] - 1) * 2

        WNmk = numpy.exp(-2j * numpy.pi * numpy.arange(N//2) / N)
        A = 0.5 * (1 - 1j * WNmk)
        B = 0.5 * (1 + 1j * WNmk)

        A_arr = plan.persistent_array(A.conj())
        B_arr = plan.persistent_array(B.conj())

        cfft_arr = Type(input_.dtype, input_.shape[:-1] + (N // 2,))
        cfft = FFT(cfft_arr, axes=(len(input_.shape) - 1,))

        prepare_output = prepare_irfft_output(cfft.parameter.output)

        cfft.parameter.output.connect(
            prepare_output, prepare_output.input, real_output=prepare_output.output)

        temp = plan.temp_array_like(cfft.parameter.input)

        batch_size = helpers.product(output.shape[:-1])

        plan.kernel_call(
            TEMPLATE.get_def('prepare_irfft_input'),
                [temp, input_, A_arr, B_arr],
                global_size=(batch_size, N // 2),
                render_kwds=dict(
                    slices=(len(input_.shape) - 1, 1),
                    N=N,
                    mul=functions.mul(input_.dtype, input_.dtype),
                    conj=functions.conj(input_.dtype)))

        plan.computation_call(cfft, output, temp, inverse=True)

        return plan


def get_multiply(output):
    return Transformation(
        [
            Parameter('output', Annotation(output, 'o')),
            Parameter('a', Annotation(output, 'i')),
            Parameter('b', Annotation(Type(output.dtype, (output.shape[-1],)), 'i'))
        ],
        """
        ${output.store_same}(${mul}(${a.load_same}, ${b.load_idx}(${idxs[-1]})));
        """,
        connectors=['output', 'a'],
        render_kwds=dict(mul=functions.mul(output.dtype, output.dtype))
        )


def get_prepare_prfft_scan(output):
    return Transformation(
        [
            Parameter('output', Annotation(output, 'o')),
            Parameter('Y', Annotation(output, 'i')),
            Parameter('re_X_0', Annotation(
                Type(dtypes.real_for(output.dtype), output.shape[:-1]), 'i'))
        ],
        """
        ${Y.ctype} Y = ${Y.load_same};
        Y = COMPLEX_CTR(${Y.ctype})(Y.y, -Y.x);

        if (${idxs[-1]} == 0)
        {
            Y.x = Y.x / 2 + ${re_X_0.load_idx}(${", ".join(idxs[:-1])});
            Y.y /= 2;
        }

        ${output.store_same}(Y);
        """,
        connectors=['output', 'Y'],
        )


class APRFFT(Computation):
    """
    FFT of a real antiperiodic signal (x[k] = -x[N/2+k]).
    """

    def __init__(self, arr_t):

        out_arr = Type(
            dtypes.complex_for(arr_t.dtype),
            arr_t.shape[:-1] + (arr_t.shape[-1] // 2,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = input_.shape[-1] * 2
        batch_shape = input_.shape[:-1]
        batch_size = helpers.product(batch_shape)

        coeffs1 = 4 * numpy.sin(2 * numpy.pi * numpy.arange(N//2) / N)
        coeffs2 = 2 * numpy.cos(2 * numpy.pi * numpy.arange(N//2) / N)

        c1_arr = plan.persistent_array(coeffs1)
        c2_arr = plan.persistent_array(coeffs2)

        multiply = get_multiply(input_)

        # re_X_1 = sum(x * coeffs2)

        t = plan.temp_array_like(input_)
        rd = Reduce(t, predicate_sum(input_.dtype), axes=(len(input_.shape)-1,))

        rd.parameter.input.connect(
            multiply, multiply.output, x=multiply.a, c2=multiply.b)

        re_X_0 = plan.temp_array_like(rd.parameter.output)
        plan.computation_call(rd, re_X_0, input_, c2_arr)

        # Y = numpy.fft.rfft(x * coeffs1)

        rfft = RFFT(input_, dont_store_last=True)
        rfft.parameter.input.connect(
            multiply, multiply.output, x=multiply.a, c1=multiply.b)

        Y = plan.temp_array_like(rfft.parameter.output)
        plan.computation_call(rfft, Y, input_, c1_arr)

        # Y *= -1j
        # Y[0] /= 2
        # Y[0] += re_X_1
        # res = numpy.cumsum(Y[:-1])

        prepare_prfft_scan = get_prepare_prfft_scan(Y)

        sc = Scan(Y, predicate_sum(Y.dtype), axes=(-1,), exclusive=False)
        sc.parameter.input.connect(
            prepare_prfft_scan, prepare_prfft_scan.output,
            Y=prepare_prfft_scan.Y, re_X_0=prepare_prfft_scan.re_X_0)

        plan.computation_call(sc, output, Y, re_X_0)

        return plan


def get_prepare_iprfft_input(X):
    # Input: size N//4
    # Output: size N//4+1

    N = X.shape[-1] * 4
    Y = Type(X.dtype, X.shape[:-1] + (N // 4 + 1,))

    return Transformation(
        [
            Parameter('Y', Annotation(Y, 'o')),
            Parameter('X', Annotation(X, 'i')),
        ],
        """
        <%
            batch_idxs = " ".join((idx + ", ") for idx in idxs[:-1])
        %>

        ${Y.ctype} Y;
        if (${idxs[-1]} == 0)
        {
            ${X.ctype} X = ${X.load_idx}(${batch_idxs} 0);
            Y = COMPLEX_CTR(${Y.ctype})(-2 * X.y, 0);
        }
        else if (${idxs[-1]} == ${N//4})
        {
            ${X.ctype} X = ${X.load_idx}(${batch_idxs} ${N//4-1});
            Y = COMPLEX_CTR(${Y.ctype})(2 * X.y, 0);
        }
        else
        {
            ${X.ctype} X = ${X.load_idx}(${batch_idxs} ${idxs[-1]});
            ${X.ctype} X_prev = ${X.load_idx}(${batch_idxs} ${idxs[-1]} - 1);
            ${X.ctype} diff = X - X_prev;
            Y = COMPLEX_CTR(${Y.ctype})(-diff.y, diff.x);
        }

        ${Y.store_same}(Y);
        """,
        connectors=['Y'],
        render_kwds=dict(N=N)
        )


def get_prepare_iprfft_output(y):
    # Input: size N//4
    # Output: size N//4

    N = y.shape[-1] * 2

    return Transformation(
        [
            Parameter('x', Annotation(y, 'o')),
            Parameter('y', Annotation(y, 'i')),
            Parameter('x0', Annotation(Type(y.dtype, y.shape[:-1]), 'i')),
            Parameter('coeffs', Annotation(Type(y.dtype, (N//2,)), 'i')),
        ],
        """
        ${y.ctype} y = ${y.load_same};
        ${coeffs.ctype} coeff = ${coeffs.load_idx}(${idxs[-1]});

        ${x.ctype} x;

        if (${idxs[-1]} == 0)
        {
            ${x0.ctype} x0 = ${x0.load_idx}(${", ".join(idxs[:-1])});
            x = x0 / ${N // 2};
        }
        else
        {
            x = y * coeff;
        }

        ${x.store_same}(x);
        """,
        connectors=['y'],
        render_kwds=dict(N=N)
        )


class IAPRFFT(Computation):
    """
    IFFT of a real antiperiodic signal (x[k] = -x[N/2+k]).
    """

    def __init__(self, arr_t):

        out_arr = Type(
            dtypes.real_for(arr_t.dtype),
            arr_t.shape[:-1] + (arr_t.shape[-1] * 2,))

        Computation.__init__(self, [
            Parameter('output', Annotation(out_arr, 'o')),
            Parameter('input', Annotation(arr_t, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):

        plan = plan_factory()

        N = input_.shape[-1] * 4
        batch_shape = input_.shape[:-1]
        batch_size = helpers.product(batch_shape)

        # The first element is unused
        coeffs = numpy.concatenate(
            [[0], 1 / (4 * numpy.sin(2 * numpy.pi * numpy.arange(1, N//2) / N))])
        coeffs_arr = plan.persistent_array(coeffs)

        prepare_iprfft_input = get_prepare_iprfft_input(input_)
        prepare_iprfft_output = get_prepare_iprfft_output(output)

        irfft = IRFFT(prepare_iprfft_input.Y)
        irfft.parameter.input.connect(
            prepare_iprfft_input, prepare_iprfft_input.Y,
            X=prepare_iprfft_input.X)
        irfft.parameter.output.connect(
            prepare_iprfft_output, prepare_iprfft_output.y,
            x=prepare_iprfft_output.x,
            x0=prepare_iprfft_output.x0, coeffs=prepare_iprfft_output.coeffs)

        real = Transformation(
            [
                Parameter('output', Annotation(Type(dtypes.real_for(input_.dtype), input_.shape), 'o')),
                Parameter('input', Annotation(input_, 'i')),
            ],
            """
            ${output.store_same}((${input.load_same}).x);
            """,
            connectors=['output']
            )

        rd_t = Type(output.dtype, input_.shape)
        rd = Reduce(rd_t, predicate_sum(rd_t.dtype), axes=(len(input_.shape)-1,))
        rd.parameter.input.connect(real, real.output, X=real.input)

        x0 = plan.temp_array_like(rd.parameter.output)

        plan.computation_call(rd, x0, input_)
        plan.computation_call(irfft, output, x0, coeffs_arr, input_)

        return plan


# Tests


def test_rfft(thr):
    N = 1024
    a = numpy.random.normal(size=N)

    rfft = RFFT(a).compile(thr)

    fa_numpy = numpy.fft.rfft(a)
    fa_ref = rfft_reference(a)

    a_dev = thr.to_device(a)
    fa_gpu = thr.empty_like(rfft.parameter.output)
    rfft(fa_gpu, a_dev)

    assert numpy.allclose(fa_numpy, fa_ref)
    assert numpy.allclose(fa_numpy, fa_gpu.get())


def test_irfft(thr):
    N = 1024
    fa = (numpy.random.normal(size=N//2+1) + 1j * numpy.random.normal(size=N//2+1))

    irfft = IRFFT(fa).compile(thr)

    a_numpy = numpy.fft.irfft(fa)
    a_ref = irfft_reference(fa)

    fa_dev = thr.to_device(fa)
    a_gpu = thr.empty_like(irfft.parameter.output)
    irfft(a_gpu, fa_dev)

    assert numpy.allclose(a_numpy, a_ref)
    assert numpy.allclose(a_numpy, a_gpu.get())


def test_aprfft(thr):
    N = 1024
    half_a = numpy.random.normal(size=N//2)
    a = numpy.concatenate([half_a, -half_a])

    aprfft = APRFFT(half_a).compile(thr)

    fa_numpy = numpy.fft.rfft(a)[1::2]
    fa_ref = aprfft_reference(half_a)

    half_a_dev = thr.to_device(half_a)
    fa_dev = thr.empty_like(aprfft.parameter.output)
    aprfft(fa_dev, half_a_dev)

    assert numpy.allclose(fa_numpy, fa_ref)
    assert numpy.allclose(fa_numpy, fa_dev.get())


def test_iaprfft(thr):
    N = 1024
    fa_odd_harmonics = (numpy.random.normal(size=N//4) + 1j * numpy.random.normal(size=N//4))

    fa = numpy.zeros(N//2+1, fa_odd_harmonics.dtype)
    fa[1::2] = fa_odd_harmonics

    iaprfft = IAPRFFT(fa_odd_harmonics).compile(thr)

    half_a_numpy = numpy.fft.irfft(fa)[:N//2]
    half_a_ref = iaprfft_reference(fa_odd_harmonics)

    fa_oh_dev = thr.to_device(fa_odd_harmonics)
    half_a_dev = thr.empty_like(iaprfft.parameter.output)
    iaprfft(half_a_dev, fa_oh_dev)

    assert numpy.allclose(half_a_numpy, half_a_ref)
    assert numpy.allclose(half_a_numpy, half_a_dev.get())


if __name__ == '__main__':
    api = any_api()
    thr = api.Thread.create(interactive=True)

    test_rfft(thr)
    test_irfft(thr)
    test_aprfft(thr)
    test_iaprfft(thr)
