"""
This example demonstrates how to implement a FFT frequency shift (``reikna.fft.FFTShift``)
as a transformation instead of a computation. A peculiarity of this transformation
is the repositioning of elements it performs (as opposed to more common
``load_same``/``store_same`` pair which keel the element order).
It makes this transformation unsafe to use for inplace kernels.

It also contains some performance tests
that compare the speed of FFT + shift as two separate computations and as a
single computation with a transformation against ``numpy`` implementation.
"""

import time
import numpy

from reikna.cluda import any_api
from reikna.fft import FFT, FFTShift

import reikna.cluda.dtypes as dtypes
from reikna.core import Transformation, Parameter, Annotation, Type


def fftshift(arr_t, axes=None):
    """
    Returns a frequency shift transformation (1 output, 1 input) that
    works as ``output = numpy.fft.fftshift(input, axes=axes)``.

    .. warning::

        Involves repositioning of the elements, so cannot be used on inplace kernels.
    """

    if axes is None:
        axes = tuple(range(len(arr_t.shape)))
    else:
        axes = tuple(sorted(axes))

    # The code taken from the FFTShift template for odd problem sizes
    # (at the moment of the writing).
    # Note the use of ``idxs`` template parameter to get access to element indices.
    return Transformation(
        [Parameter('output', Annotation(arr_t, 'o')),
        Parameter('input', Annotation(arr_t, 'i'))],
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
        connectors=['input'],
        render_kwds=dict(axes=axes))


def run_test(thr, shape, dtype, axes=None):

    data = numpy.random.normal(size=shape).astype(dtype)

    fft = FFT(data, axes=axes)
    fftc = fft.compile(thr)

    shift = FFTShift(data, axes=axes)
    shiftc = shift.compile(thr)

    # FFT + shift as two separate computations

    data_dev = thr.to_device(data)

    t_start = time.time()
    fftc(data_dev, data_dev)
    thr.synchronize()
    t_gpu_fft = time.time() - t_start

    t_start = time.time()
    shiftc(data_dev, data_dev)
    thr.synchronize()
    t_gpu_shift = time.time() - t_start

    data_dev = thr.to_device(data)

    t_start = time.time()
    fftc(data_dev, data_dev)
    shiftc(data_dev, data_dev)
    thr.synchronize()
    t_gpu_separate = time.time() - t_start

    data_gpu = data_dev.get()

    # FFT + shift as a computation with a transformation

    data_dev = thr.to_device(data)

    # a separate output array to avoid unsafety of the shift transformation
    res_dev = thr.empty_like(data_dev)

    shift_tr = fftshift(data, axes=axes)
    fft2 = fft.parameter.output.connect(shift_tr, shift_tr.input, new_output=shift_tr.output)
    fft2c = fft2.compile(thr)

    t_start = time.time()
    fft2c(res_dev, data_dev)
    thr.synchronize()
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

    data_gpu2 = res_dev.get()

    # Checking that the results are correct
    # (note: this will require relaxing the tolerances
    # if complex64 is used instead of complex128)
    assert numpy.allclose(data_ref, data_gpu)
    assert numpy.allclose(data_ref, data_gpu2)

    return dict(
        t_gpu_fft=t_gpu_fft,
        t_gpu_shift=t_gpu_shift,
        t_gpu_separate=t_gpu_separate,
        t_gpu_combined=t_gpu_combined,
        t_cpu_fft=t_cpu_fft,
        t_cpu_shift=t_cpu_shift,
        t_cpu_all=t_cpu_all)


def run_tests(thr, shape, dtype, axes=None, attempts=10):
    results = [run_test(thr, shape, dtype, axes=axes) for i in range(attempts)]
    return {key:min(result[key] for result in results) for key in results[0]}

if __name__ == '__main__':
    api = any_api()
    thr = api.Thread.create()

    shape = (1024, 1024)
    dtype = numpy.complex128
    axes = (0, 1)

    results = run_tests(thr, shape, dtype, axes=axes)

    print('device:', thr._device.name)
    print('shape:', shape)
    print('dtype:', dtype)
    print('axes:', axes)

    for key, val in results.items():
        print(key, ':', val)

    print(
        "Speedup for a separate calculation:",
        results['t_cpu_all'] / results['t_gpu_separate'])

    print(
        "Speedup for a combined calculation:",
        results['t_cpu_all'] / results['t_gpu_combined'])

    print(
        "Speedup for fft alone:",
        results['t_cpu_fft'] / results['t_gpu_fft'])

    print(
        "Speedup for shift alone:",
        results['t_cpu_shift'] / results['t_gpu_shift'])
