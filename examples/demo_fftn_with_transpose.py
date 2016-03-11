"""
This example shows how to implement an n-dimensional FFT over arbitrary axes
using a 1D FFT over the innermost dimension and transpositions.

At the moment of the writing, testing shows that the performance of this approach
is generally worse than that of the existing global FFT kernel
(the worst case being a 3D FFT with dimensions 1024x(16, 16, 16) on CUDA,
where the performance is almost halved).

Nevertheless, this computation is preserved as an example in case it is ever needed,
because it will greatly simplify the FFT computation.
"""

import time

import numpy

from reikna.cluda import any_api
from reikna.core import Computation, Parameter, Annotation
from reikna.fft import FFT
from reikna.algorithms import Transpose


class FFTWithTranspose(Computation):

    def __init__(self, arr_t, axes=None):

        Computation.__init__(self, [
            Parameter('output', Annotation(arr_t, 'o')),
            Parameter('input', Annotation(arr_t, 'i')),
            Parameter('inverse', Annotation(numpy.int32), default=0)])

        if axes is None:
            axes = range(len(arr_t.shape))
        self._axes = tuple(sorted(axes))


    def _build_plan(self, plan_factory, device_params, output, input_, inverse):

        plan = plan_factory()

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
                    + local_axes[current_axis+1:]
                    + [local_axes[current_axis]])

                # That's the corresponding permutation of the original axes
                # (we need to keep track of it)
                current_axes = (
                    current_axes[:current_axis]
                    + current_axes[current_axis+1:]
                    + [current_axes[current_axis]])

                # Transpose the array, saving the result in a temporary buffer
                tr = Transpose(current_input, local_axes)
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
            pairs = list(zip(current_axes, list(range(num_axes))))
            tr_axes = [local_axis for _, local_axis in sorted(pairs)]
            tr = Transpose(current_output, tr_axes)
            plan.computation_call(tr, output, current_output)

        return plan


if __name__ == '__main__':

    api = any_api()
    thr = api.Thread.create()

    dtype = numpy.complex128

    shape = (1024, 16, 16, 16)
    axes = (1, 2, 3)

    data = numpy.random.normal(size=shape) + 1j * numpy.random.normal(size=shape)
    data = data.astype(dtype)

    fft = FFT(data, axes=axes)
    fftc = fft.compile(thr)

    fft2 = FFTWithTranspose(data, axes=axes)
    fft2c = fft2.compile(thr)

    data_dev = thr.to_device(data)
    res_dev = thr.empty_like(data_dev)

    for comp, tag in [(fftc, "original FFT"), (fft2c, "transposition-based FFT")]:
        attempts = 10
        ts = []
        for i in range(attempts):
            t1 = time.time()
            comp(res_dev, data_dev)
            thr.synchronize()
            t2 = time.time()
            ts.append(t2 - t1)

        fwd_ref = numpy.fft.fftn(data, axes=axes).astype(dtype)
        assert numpy.allclose(res_dev.get(), fwd_ref)

        print(tag, min(ts), "s")
