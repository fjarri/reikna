"""
An example demonstrating how to implement a FFT frequency shift (``reikna.fft.FFTShift``)
as a transformation instead of a computation. A peculiarity of this transformation
is the repositioning of elements it performs (as opposed to more common
``load_same``/``store_same`` pair which keel the element order).
It makes this transformation unsafe to use for inplace kernels.

It also contains some performance tests
that compare the speed of FFT + shift as two separate computations and as a
single computation with a transformation against ``numpy`` implementation.
"""

import time
from collections.abc import Sequence

import numpy
from grunnur import API, Array, AsArrayMetadata, Context, Queue
from numpy.typing import DTypeLike

from reikna.core import Annotation, Parameter, Transformation, Type
from reikna.fft import FFT, FFTShift


def fftshift(arr: AsArrayMetadata, axes: None | Sequence[int] = None) -> Transformation:
    """
    Returns a frequency shift transformation (1 output, 1 input) that
    works as ``output = numpy.fft.fftshift(input, axes=axes)``.

    .. warning::

        Involves repositioning of the elements, so cannot be used on inplace kernels.
    """
    metadata = arr.as_array_metadata()
    normalized_axes = tuple(range(len(metadata.shape)) if axes is None else sorted(axes))

    # The code taken from the FFTShift template for odd problem sizes
    # (at the moment of the writing).
    # Note the use of ``idxs`` template parameter to get access to element indices.
    return Transformation(
        [
            Parameter("output", Annotation(metadata, "o")),
            Parameter("input", Annotation(metadata, "i")),
        ],
        """
        <%
            dimensions = len(output.shape)
            new_idx_names = ['new_idx' + str(i) for i in range(dimensions)]
        %>
        %for dim in range(dimensions):
        VSIZE_T ${new_idx_names[dim]} =
            ${idxs[dim]}
            %if dim in axes:
                %if output.shape[dim] % 2 == 0:
                + (${idxs[dim]} < ${output.shape[dim] // 2} ?
                    ${output.shape[dim] // 2} :
                    ${-output.shape[dim] // 2})
                %else:
                + (${idxs[dim]} <= ${output.shape[dim] // 2} ?
                    ${output.shape[dim] // 2} :
                    ${-(output.shape[dim] // 2 + 1)})
                %endif
            %endif
            ;
        %endfor

        ${output.ctype} val = ${input.load_same};
        ${output.store_idx}(${', '.join(new_idx_names)}, val);
        """,
        connectors=["input"],
        render_kwds=dict(axes=normalized_axes),
    )


def run_test(
    queue: Queue, shape: tuple[int, ...], dtype: DTypeLike, axes: None | Sequence[int] = None
) -> dict[str, float]:
    data = numpy.random.default_rng().normal(size=shape).astype(dtype)
    data_dev = Array.from_host(queue.device, data)

    fft = FFT(data_dev, axes=axes)
    fftc = fft.compile(queue.device)

    shift = FFTShift(data_dev, axes=axes)
    shiftc = shift.compile(queue.device)

    # FFT + shift as two separate computations

    t_start = time.time()
    fftc(queue, data_dev, data_dev)
    queue.synchronize()
    t_gpu_fft = time.time() - t_start

    t_start = time.time()
    shiftc(queue, data_dev, data_dev)
    queue.synchronize()
    t_gpu_shift = time.time() - t_start

    data_dev = Array.from_host(queue.device, data)

    t_start = time.time()
    fftc(queue, data_dev, data_dev)
    shiftc(queue, data_dev, data_dev)
    queue.synchronize()
    t_gpu_separate = time.time() - t_start

    data_gpu = data_dev.get(queue)

    # FFT + shift as a computation with a transformation

    data_dev = Array.from_host(queue, data)

    # a separate output array to avoid unsafety of the shift transformation
    res_dev = Array.empty_like(queue.device, data_dev)

    shift_tr = fftshift(data_dev, axes=axes)
    fft2 = fft.parameter.output.connect(
        shift_tr, shift_tr.parameter.input, new_output=shift_tr.parameter.output
    )
    fft2c = fft2.compile(queue.device)

    t_start = time.time()
    fft2c(queue, res_dev, data_dev)
    queue.synchronize()
    t_gpu_combined = time.time() - t_start

    # Reference calculation with numpy

    t_start = time.time()
    numpy.fft.fftn(data, axes=axes)
    t_cpu_fft = time.time() - t_start

    t_start = time.time()
    numpy.fft.fftshift(data, axes=axes)
    t_cpu_shift = time.time() - t_start

    t_start = time.time()
    data_ref = numpy.fft.fftn(data, axes=axes)
    data_ref = numpy.fft.fftshift(data_ref, axes=axes)
    t_cpu_all = time.time() - t_start

    data_gpu2 = res_dev.get(queue)

    # Checking that the results are correct
    # (note: this will require relaxing the tolerances
    # if complex64 is used instead of complex128)
    assert numpy.allclose(data_ref, data_gpu, atol=1e-2, rtol=1e-4)  # noqa: S101
    assert numpy.allclose(data_ref, data_gpu2, atol=1e-2, rtol=1e-4)  # noqa: S101

    return dict(
        t_gpu_fft=t_gpu_fft,
        t_gpu_shift=t_gpu_shift,
        t_gpu_separate=t_gpu_separate,
        t_gpu_combined=t_gpu_combined,
        t_cpu_fft=t_cpu_fft,
        t_cpu_shift=t_cpu_shift,
        t_cpu_all=t_cpu_all,
    )


def run_tests(
    queue: Queue,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    axes: None | Sequence[int] = None,
    attempts: int = 10,
) -> dict[str, float]:
    results = [run_test(queue, shape, dtype, axes=axes) for i in range(attempts)]
    return {key: min(result[key] for result in results) for key in results[0]}


if __name__ == "__main__":
    context = Context.from_devices([API.any().platforms[0].devices[0]])
    queue = Queue(context.device)

    shape = (1024, 1024)
    dtype = numpy.complex64
    axes = (0, 1)

    results = run_tests(queue, shape, dtype, axes=axes)

    print("device:", queue.device.name)  # noqa: T201
    print("shape:", shape)  # noqa: T201
    print("dtype:", dtype)  # noqa: T201
    print("axes:", axes)  # noqa: T201

    for key, val in results.items():
        print(key, ":", val)  # noqa: T201

    print("Speedup for a separate calculation:", results["t_cpu_all"] / results["t_gpu_separate"])  # noqa: T201
    print("Speedup for a combined calculation:", results["t_cpu_all"] / results["t_gpu_combined"])  # noqa: T201
    print("Speedup for fft alone:", results["t_cpu_fft"] / results["t_gpu_fft"])  # noqa: T201
    print("Speedup for shift alone:", results["t_cpu_shift"] / results["t_gpu_shift"])  # noqa: T201
