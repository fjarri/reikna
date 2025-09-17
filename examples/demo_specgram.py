"""
An Texample demonstrating how to write an analogue of the ``matplotlib.mlab.specgram``
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

from collections.abc import Callable
from typing import Any, cast

import matplotlib
import numpy
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grunnur import (
    API,
    Array,
    ArrayMetadata,
    AsArrayMetadata,
    Context,
    DeviceParameters,
    Queue,
    dtypes,
    functions,
)
from matplotlib.mlab import specgram, window_hanning

from reikna import transformations
from reikna.algorithms import Transpose
from reikna.core import (
    Annotation,
    Computation,
    ComputationPlan,
    KernelArguments,
    Parameter,
    Transformation,
    Type,
)
from reikna.fft import FFT


def get_data() -> tuple[NDArray[numpy.float64], dict[str, Any]]:
    rng = numpy.random.default_rng()

    dt = 0.0005
    t = numpy.arange(0.0, 20.0, dt)
    s1 = numpy.sin(2 * numpy.pi * 100 * t)
    s2 = 2 * numpy.sin(2 * numpy.pi * 400 * t)

    # create a transient "chirp"
    mask = numpy.where(numpy.logical_and(t > 10, t < 12), 1.0, 0.0)  # noqa: PLR2004
    s2 = s2 * mask

    # add some noise into the mix
    nse = 0.01 * rng.standard_normal(len(t))

    x = s1 + s2 + nse  # the signal
    fft_size = 1024  # the length of the windowing segments
    freqs = int(1.0 / dt)  # the sampling frequency

    return x.astype(numpy.float32), dict(NFFT=fft_size, Fs=freqs, noverlap=900, pad_to=2048)


def hanning_window(arr: AsArrayMetadata, fft_size: int) -> Transformation:
    """
    Applies the von Hann window to the rows of a 2D array.
    To account for zero padding (which we do not want to window), fft_size is provided separately.
    """
    metadata = arr.as_array_metadata()
    if dtypes.is_complex(metadata.dtype):
        coeff_dtype = dtypes.real_for(metadata.dtype)
    else:
        coeff_dtype = metadata.dtype
    return Transformation(
        [
            Parameter("output", Annotation(metadata, "o")),
            Parameter("input", Annotation(metadata, "i")),
        ],
        """
        ${dtypes.ctype(coeff_dtype)} coeff;
        %if fft_size != output.shape[0]:
        if (${idxs[1]} >= ${fft_size})
        {
            coeff = 1;
        }
        else
        %endif
        {
            coeff = 0.5 * (1 - cos(2 * ${numpy.pi} * ${idxs[-1]} / (${fft_size} - 1)));
        }
        ${output.store_same}(${mul}(${input.load_same}, coeff));
        """,
        render_kwds=dict(
            dtypes=dtypes,
            coeff_dtype=coeff_dtype,
            fft_size=fft_size,
            mul=functions.mul(metadata.dtype, coeff_dtype),
        ),
    )


def rolling_frame(
    arr: AsArrayMetadata, fft_size: int, noverlap: int, pad_to: int | None
) -> Transformation:
    """
    Transforms a 1D array to a 2D array whose rows are
    partially overlapped parts of the initial array.
    """
    metadata = arr.as_array_metadata()

    assert len(metadata.shape) == 1  # noqa: S101

    frame_step = fft_size - noverlap
    frame_num = (metadata.shape[0] - noverlap) // frame_step
    frame_size = fft_size if pad_to is None else pad_to

    result_arr = ArrayMetadata((frame_num, frame_size), metadata.dtype)

    return Transformation(
        [
            Parameter("output", Annotation(result_arr, "o")),
            Parameter("input", Annotation(metadata, "i")),
        ],
        """
        %if fft_size != output.shape[1]:
        if (${idxs[1]} >= ${fft_size})
        {
            ${output.store_same}(0);
        }
        else
        %endif
        {
            ${output.store_same}(${input.load_idx}(${idxs[0]} * ${frame_step} + ${idxs[1]}));
        }
        """,
        render_kwds=dict(frame_step=frame_step, fft_size=fft_size),
        # note that only the "store_same"-using argument can serve as a connector!
        connectors=["output"],
    )


def crop_frequencies(arr: AsArrayMetadata) -> Transformation:
    """
    Crop a 2D array whose columns represent frequencies to only leave the frequencies with
    different absolute values.
    """
    metadata = arr.as_array_metadata()
    result_arr = ArrayMetadata((metadata.shape[0], metadata.shape[1] // 2 + 1), metadata.dtype)
    return Transformation(
        [
            Parameter("output", Annotation(result_arr, "o")),
            Parameter("input", Annotation(metadata, "i")),
        ],
        """
        if (${idxs[1]} < ${input.shape[1] // 2 + 1})
            ${output.store_idx}(${idxs[0]}, ${idxs[1]}, ${input.load_same});
        """,
        # note that only the "load_same"-using argument can serve as a connector!
        connectors=["input"],
    )


class Spectrogram(Computation):
    def __init__(
        self,
        x: AsArrayMetadata,
        fft_size: int = 256,
        noverlap: int = 128,
        pad_to: int | None = None,
        window: Callable[[AsArrayMetadata, int], Transformation] = hanning_window,
    ):
        metadata = x.as_array_metadata()
        assert dtypes.is_real(metadata.dtype)  # noqa: S101
        assert len(metadata.shape) == 1  # noqa: S101

        rolling_frame_trf = rolling_frame(metadata, fft_size, noverlap, pad_to)

        complex_dtype = dtypes.complex_for(metadata.dtype)
        fft_arr = ArrayMetadata(rolling_frame_trf.parameter.output.shape, complex_dtype)
        real_fft_arr = ArrayMetadata(rolling_frame_trf.parameter.output.shape, metadata.dtype)

        window_trf = window(real_fft_arr, fft_size)
        broadcast_zero_trf = transformations.broadcast_const(real_fft_arr, 0)
        to_complex_trf = transformations.combine_complex(fft_arr)
        amplitude_trf = transformations.norm_const(fft_arr, 1)
        crop_trf = crop_frequencies(amplitude_trf.parameter.output)

        fft = FFT(fft_arr, axes=(1,))
        fft.parameter.input.connect(
            to_complex_trf,
            to_complex_trf.parameter.output,
            input_real=to_complex_trf.parameter.real,
            input_imag=to_complex_trf.parameter.imag,
        )
        fft.parameter.input_imag.connect(broadcast_zero_trf, broadcast_zero_trf.parameter.output)
        fft.parameter.input_real.connect(
            window_trf, window_trf.parameter.output, unwindowed_input=window_trf.parameter.input
        )
        fft.parameter.unwindowed_input.connect(
            rolling_frame_trf,
            rolling_frame_trf.parameter.output,
            flat_input=rolling_frame_trf.parameter.input,
        )
        fft.parameter.output.connect(
            amplitude_trf, amplitude_trf.parameter.input, amplitude=amplitude_trf.parameter.output
        )
        fft.parameter.amplitude.connect(
            crop_trf, crop_trf.parameter.input, cropped_amplitude=crop_trf.parameter.output
        )

        self._fft = fft

        self._transpose = Transpose(fft.parameter.cropped_amplitude)

        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(self._transpose.parameter.output, "o")),
                Parameter("input", Annotation(fft.parameter.flat_input, "i")),
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
        temp = plan.temp_array_like(self._fft.parameter.cropped_amplitude)
        plan.computation_call(self._fft, temp, input_)
        plan.computation_call(self._transpose, output, temp)
        return plan


if __name__ == "__main__":
    x, params = get_data()

    fig = plt.figure()
    s = fig.add_subplot(2, 1, 1)
    spectre, freqs, ts = specgram(x, mode="magnitude", **params)
    window = window_hanning(numpy.ones(params["NFFT"]))

    assert isinstance(ts, numpy.ndarray)  # noqa: S101
    assert isinstance(freqs, numpy.ndarray)  # noqa: S101
    assert isinstance(window, numpy.ndarray)  # noqa: S101

    # Renormalize to match the computation
    spectre *= window.sum()

    s.imshow(
        numpy.log10(spectre),
        extent=(ts[0], ts[-1], freqs[0], freqs[-1]),
        aspect="auto",
        origin="lower",
    )

    context = Context.from_devices([API.any().platforms[0].devices[0]])
    queue = Queue(context.device)

    x_dev = Array.from_host(queue, x)

    specgram_reikna = Spectrogram(
        x_dev, fft_size=params["NFFT"], noverlap=params["noverlap"], pad_to=params["pad_to"]
    ).compile(queue.device)

    spectre_dev = Array.empty_like(queue.device, specgram_reikna.parameter.output)
    specgram_reikna(queue, spectre_dev, x_dev)
    spectre_reikna = spectre_dev.get(queue)

    assert numpy.allclose(spectre, spectre_reikna, atol=1e-4, rtol=1e-4)  # noqa: S101

    s = fig.add_subplot(2, 1, 2)
    im = s.imshow(
        numpy.log10(spectre_reikna),
        extent=(ts[0], ts[-1], freqs[0], freqs[-1]),
        aspect="auto",
        origin="lower",
    )
    fig.savefig("demo_specgram.png")
