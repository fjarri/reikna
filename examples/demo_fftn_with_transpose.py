"""
An example showing how to implement an n-dimensional FFT over arbitrary axes
using a 1D FFT over the innermost dimension and transpositions.

At the moment of the writing, testing shows that the performance of this approach
is generally worse than that of the existing global FFT kernel
(the worst case being a 3D FFT with dimensions 1024x(16, 16, 16) on CUDA,
where the performance is almost halved).

Nevertheless, this computation is preserved as an example in case it is ever needed,
because it will greatly simplify the FFT computation.
"""

import time
from collections.abc import Callable, Sequence

import numpy
from grunnur import API, Array, AsArrayMetadata, Context, DeviceParameters, Queue

from reikna.algorithms import Transpose
from reikna.core import Annotation, Computation, ComputationPlan, KernelArguments, Parameter
from reikna.fft import FFT


class FFTWithTranspose(Computation):
    def __init__(self, arr: AsArrayMetadata, axes: None | Sequence[int] = None):
        metadata = arr.as_array_metadata()
        Computation.__init__(
            self,
            [
                Parameter("output", Annotation(metadata, "o")),
                Parameter("input", Annotation(metadata, "i")),
                Parameter("inverse", Annotation(numpy.int32), default=0),
            ],
        )

        if axes is None:
            axes = range(len(metadata.shape))
        self._axes = tuple(sorted(axes))

    def _build_plan(
        self,
        plan_factory: Callable[[], ComputationPlan],
        _device_params: DeviceParameters,
        args: KernelArguments,
    ) -> ComputationPlan:
        plan = plan_factory()

        output = args.output
        input_ = args.input
        inverse = args.inverse

        num_axes = len(input_.shape)

        current_axes = list(range(num_axes))
        current_input = input_

        # Iterate over all the axes we need to FFT over
        for i, initial_axis in enumerate(self._axes):
            # Find out where the target axis is currently located
            current_axis = current_axes.index(initial_axis)

            # If it is not the innermost one, we will transpose the array
            # to bring it to the end.
            if current_axis != len(current_axes) - 1:
                local_axes = list(range(num_axes))

                # The `Transpose` computation is most efficient when we ask it
                # to swap two innermost parts of the axes list, e.g.
                # [0, 1, 2, 3, 4, 5] to [0, 1, 4, 5, 2, 3]
                # This way the transposition requires only one kernel call.

                # Since we do not care where the other axes go as long as the target one
                # becomes the innermost one, it is easy to follow this guideline.

                # That's the transposition that we will need to perform
                # on the current array
                local_axes = (
                    local_axes[:current_axis]
                    + local_axes[current_axis + 1 :]
                    + [local_axes[current_axis]]
                )

                # That's the corresponding permutation of the original axes
                # (we need to keep track of it)
                current_axes = (
                    current_axes[:current_axis]
                    + current_axes[current_axis + 1 :]
                    + [current_axes[current_axis]]
                )

                # Transpose the array, saving the result in a temporary buffer
                tr = Transpose(current_input, axes=local_axes)
                temp = plan.temp_array_like(tr.parameter.output)
                plan.computation_call(tr, temp, current_input)

                # Now the target axis is the innermost one
                current_axis = len(current_axes) - 1
                current_input = temp

            # If it is the last FFT to perform, and there is no final transposition required,
            # the FFT output is pointed to the output array instead of a temporary buffer.
            if i == len(self._axes) - 1 and current_axes == list(range(num_axes)):
                current_output = output
            else:
                current_output = current_input

            fft = FFT(current_input, axes=(current_axis,))
            plan.computation_call(fft, current_output, current_input, inverse)

        # If the axes are not in the original order, there is one last transposition required
        if current_axes != list(range(num_axes)):
            pairs = list(zip(current_axes, list(range(num_axes)), strict=True))
            tr_axes = [local_axis for _, local_axis in sorted(pairs)]
            tr = Transpose(current_output, axes=tr_axes)
            plan.computation_call(tr, output, current_output)

        return plan


if __name__ == "__main__":
    context = Context.from_devices([API.any().platforms[0].devices[0]])
    queue = Queue(context.device)

    dtype = numpy.complex64

    shape = (1024, 16, 16, 16)
    axes = (1, 2, 3)

    rng = numpy.random.default_rng()
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    data = data.astype(dtype)

    data_dev = Array.from_host(queue.device, data)
    res_dev = Array.empty_like(queue.device, data_dev)

    fft = FFT(data_dev, axes=axes)
    fftc = fft.compile(queue.device)

    fft2 = FFTWithTranspose(data_dev, axes=axes)
    fft2c = fft2.compile(queue.device)

    for comp, tag in [(fftc, "original FFT"), (fft2c, "transposition-based FFT")]:
        attempts = 10
        ts = []
        for _ in range(attempts):
            t1 = time.time()
            comp(queue, res_dev, data_dev)
            queue.synchronize()
            t2 = time.time()
            ts.append(t2 - t1)

        fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
        assert numpy.allclose(res_dev.get(queue), fwd_ref, atol=1e-4, rtol=1e-4)  # noqa: S101

        print(tag, min(ts), "s")  # noqa: T201
