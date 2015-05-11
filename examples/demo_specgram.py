"""
This example demonstrates how to write an analogue of the ``matplotlib.mlab.specgram``
function using Reikna. The main difficulty here is to split the task into transformations
and computation cores and assemble them into a single computation.

To map the task to the transformation/computation model, one needs to separate the steps
into those that do not require the communication between threads, and those that do.
The former will be transformations, the latter will be computations.

``specgram`` does the following:

* Reshapes the initial 1D array into a 2D with an overlap between rows;
* Applies a window function;
* Performs an FFT over the rows;
* Crops the negative frequencies part of the result (default behavior);
* Calculates the spectrum. The actual formula here depends on the ``mode`` parameter.
  This example uses ``mode='amplitude'`` to make the example simpler,
  because the default ``mode='psd'`` does a bit of additional scaling.
  Implementing other modes is left as an exercise for the reader.

Of all these, only FFT is a computation, the rest can be implemented as transformations.

Note that ``specgram`` returns the array of shape ``(frequencies, times)``.
Reikna's batched FFT is most effective when performed over the last axis of the array,
so we will operate with ``(times, frequencies)`` array and transpose it in the end.
Transposition is also a computation, so we will have to chain it after the FFT.

Since the FFT requires a complex-valued array, we will need an additional transformation that
typecasts the initial real-valued array into complex numbers. One could write a custom
computation that does that, but we will use two predefined transformations
(from the ``reikna.transformations`` module): ``combine_complex`` (creates a complex array
out of real and imaginary parts) and ``broadcast_const`` (to fill the complex part with zeroes).

CAVEAT: In the steps above, two of the transformations (the initial reshape and the frequency crop)
will change the shape of the array. Currently such transformations can only be leaves of the
transformation tree, so we will have to swap the cropping step and the spectrum calculating step.
"""

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram

from reikna.cluda import any_api
from reikna.cluda import dtypes, functions
from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.fft import FFT
from reikna.algorithms import Transpose
import reikna.transformations as transformations


def get_data():

    dt = 0.0005
    t = numpy.arange(0.0, 20.0, dt)
    s1 = numpy.sin(2*numpy.pi*100*t)
    s2 = 2*numpy.sin(2*numpy.pi*400*t)

    # create a transient "chirp"
    mask = numpy.where(numpy.logical_and(t > 10, t < 12), 1.0, 0.0)
    s2 = s2 * mask

    # add some noise into the mix
    nse = 0.01*numpy.random.randn(len(t))

    x = s1 + s2 + nse # the signal
    NFFT = 1024       # the length of the windowing segments
    Fs = int(1.0/dt)  # the sampling frequency

    return x, dict(NFFT=NFFT, Fs=Fs, noverlap=900, pad_to=2048)


def hanning_window(arr, NFFT):
    """
    Applies the von Hann window to the rows of a 2D array.
    To account for zero padding (which we do not want to window), NFFT is provided separately.
    """
    if dtypes.is_complex(arr.dtype):
        coeff_dtype = dtypes.real_for(arr.dtype)
    else:
        coeff_dtype = arr.dtype
    return Transformation(
        [
            Parameter('output', Annotation(arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        ${dtypes.ctype(coeff_dtype)} coeff;
        %if NFFT != output.shape[0]:
        if (${idxs[1]} >= ${NFFT})
        {
            coeff = 1;
        }
        else
        %endif
        {
            coeff = 0.5 * (1 - cos(2 * ${numpy.pi} * ${idxs[-1]} / (${NFFT} - 1)));
        }
        ${output.store_same}(${mul}(${input.load_same}, coeff));
        """,
        render_kwds=dict(
            coeff_dtype=coeff_dtype, NFFT=NFFT,
            mul=functions.mul(arr.dtype, coeff_dtype)))


def rolling_frame(arr, NFFT, noverlap, pad_to):
    """
    Transforms a 1D array to a 2D array whose rows are
    partially overlapped parts of the initial array.
    """

    frame_step = NFFT - noverlap
    frame_num = (arr.size - noverlap) // frame_step
    frame_size = NFFT if pad_to is None else pad_to

    result_arr = Type(arr.dtype, (frame_num, frame_size))

    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        %if NFFT != output.shape[1]:
        if (${idxs[1]} >= ${NFFT})
        {
            ${output.store_same}(0);
        }
        else
        %endif
        {
            ${output.store_same}(${input.load_idx}(${idxs[0]} * ${frame_step} + ${idxs[1]}));
        }
        """,
        render_kwds=dict(frame_step=frame_step, NFFT=NFFT),
        # note that only the "store_same"-using argument can serve as a connector!
        connectors=['output'])


def crop_frequencies(arr):
    """
    Crop a 2D array whose columns represent frequencies to only leave the frequencies with
    different absolute values.
    """
    result_arr = Type(arr.dtype, (arr.shape[0], arr.shape[1] // 2 + 1))
    return Transformation(
        [
            Parameter('output', Annotation(result_arr, 'o')),
            Parameter('input', Annotation(arr, 'i')),
        ],
        """
        if (${idxs[1]} < ${input.shape[1] // 2 + 1})
            ${output.store_idx}(${idxs[0]}, ${idxs[1]}, ${input.load_same});
        """,
        # note that only the "load_same"-using argument can serve as a connector!
        connectors=['input'])


class Spectrogram(Computation):

    def __init__(self, x, NFFT=256, noverlap=128, pad_to=None, window=hanning_window):

        assert dtypes.is_real(x.dtype)
        assert x.ndim == 1

        rolling_frame_trf = rolling_frame(x, NFFT, noverlap, pad_to)

        complex_dtype = dtypes.complex_for(x.dtype)
        fft_arr = Type(complex_dtype, rolling_frame_trf.output.shape)
        real_fft_arr = Type(x.dtype, rolling_frame_trf.output.shape)

        window_trf = window(real_fft_arr, NFFT)
        broadcast_zero_trf = transformations.broadcast_const(real_fft_arr, 0)
        to_complex_trf = transformations.combine_complex(fft_arr)
        amplitude_trf = transformations.norm_const(fft_arr, 1)
        crop_trf = crop_frequencies(amplitude_trf.output)

        fft = FFT(fft_arr, axes=(1,))
        fft.parameter.input.connect(
            to_complex_trf, to_complex_trf.output,
            input_real=to_complex_trf.real, input_imag=to_complex_trf.imag)
        fft.parameter.input_imag.connect(
            broadcast_zero_trf, broadcast_zero_trf.output)
        fft.parameter.input_real.connect(
            window_trf, window_trf.output, unwindowed_input=window_trf.input)
        fft.parameter.unwindowed_input.connect(
            rolling_frame_trf, rolling_frame_trf.output, flat_input=rolling_frame_trf.input)
        fft.parameter.output.connect(
            amplitude_trf, amplitude_trf.input, amplitude=amplitude_trf.output)
        fft.parameter.amplitude.connect(
            crop_trf, crop_trf.input, cropped_amplitude=crop_trf.output)

        self._fft = fft

        self._transpose = Transpose(fft.parameter.cropped_amplitude)

        Computation.__init__(self,
            [Parameter('output', Annotation(self._transpose.parameter.output, 'o')),
            Parameter('input', Annotation(fft.parameter.flat_input, 'i'))])

    def _build_plan(self, plan_factory, device_params, output, input_):
        plan = plan_factory()
        temp = plan.temp_array_like(self._fft.parameter.cropped_amplitude)
        plan.computation_call(self._fft, temp, input_)
        plan.computation_call(self._transpose, output, temp)
        return plan


if __name__ == '__main__':

    numpy.random.seed(125)
    x, params = get_data()

    fig = plt.figure()
    s = fig.add_subplot(2, 1, 1)
    spectre, freqs, ts = specgram(x, mode='magnitude', **params)
    s.imshow(
        numpy.log10(spectre),
        extent=(ts[0], ts[-1], freqs[0], freqs[-1]),
        aspect='auto',
        origin='lower')

    api = any_api()
    thr = api.Thread.create()

    specgram_reikna = Spectrogram(
        x, NFFT=params['NFFT'], noverlap=params['noverlap'], pad_to=params['pad_to']).compile(thr)

    x_dev = thr.to_device(x)
    spectre_dev = thr.empty_like(specgram_reikna.parameter.output)
    specgram_reikna(spectre_dev, x_dev)
    spectre_reikna = spectre_dev.get()

    assert numpy.allclose(spectre, spectre_reikna)

    s = fig.add_subplot(2, 1, 2)
    im=s.imshow(
        numpy.log10(spectre_reikna),
        extent=(ts[0], ts[-1], freqs[0], freqs[-1]),
        aspect='auto',
        origin='lower')
    fig.savefig('demo_specgram.png')
