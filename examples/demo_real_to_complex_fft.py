"""
This example illustrates how to:
- attach a transformation to an FFT computation object that will make it
  operate on real-valued inputs.
"""

import numpy
from grunnur import Array, Context, Queue, any_api, dtypes

from reikna.core import Annotation, Parameter, Transformation, Type
from reikna.fft import FFT

# Pick the first available GPGPU API and make a queue on it.
context = Context.from_devices([any_api.platforms[0].devices[0]])
queue = Queue(context.device)


# A transformation that transforms a real array to a complex one
# by adding a zero imaginary part
def get_complex_trf(arr):
    complex_dtype = dtypes.complex_for(arr.dtype)
    return Transformation(
        [
            Parameter("output", Annotation(Type.array(complex_dtype, arr.shape), "o")),
            Parameter("input", Annotation(arr, "i")),
        ],
        """
        ${output.store_same}(
            COMPLEX_CTR(${output.ctype})(
                ${input.load_same},
                0));
        """,
    )


arr = numpy.random.normal(size=3000).astype(numpy.float32)

trf = get_complex_trf(arr)


# Create the FFT computation and attach the transformation above to its input.
fft = FFT(trf.output)  # (A shortcut: using the array type saved in the transformation)
fft.parameter.input.connect(trf, trf.output, new_input=trf.input)
cfft = fft.compile(queue.device)


# Run the computation
arr_dev = Array.from_host(queue, arr)
res_dev = Array.empty(queue.device, arr.shape, numpy.complex64)
cfft(queue, res_dev, arr_dev)
result = res_dev.get(queue)

reference = numpy.fft.fft(arr)

assert numpy.linalg.norm(result - reference) / numpy.linalg.norm(reference) < 1e-6
