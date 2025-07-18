"""
An example illustrating how to:
- attach a transformation to an FFT computation object that will make it
  operate on real-valued inputs.
"""

import numpy
from grunnur import API, Array, ArrayMetadata, AsArrayMetadata, Context, Queue, dtypes

from reikna.core import Annotation, Parameter, Transformation, Type
from reikna.fft import FFT

# Pick the first available GPGPU API and make a queue on it.
context = Context.from_devices([API.any().platforms[0].devices[0]])
queue = Queue(context.device)


# A transformation that transforms a real array to a complex one
# by adding a zero imaginary part
def get_complex_trf(arr: AsArrayMetadata) -> Transformation:
    metadata = arr.as_array_metadata()
    complex_dtype = dtypes.complex_for(metadata.dtype)
    return Transformation(
        [
            Parameter("output", Annotation(ArrayMetadata(metadata.shape, complex_dtype), "o")),
            Parameter("input", Annotation(metadata, "i")),
        ],
        """
        ${output.store_same}(
            COMPLEX_CTR(${output.ctype})(
                ${input.load_same},
                0));
        """,
    )


arr = numpy.random.default_rng().normal(size=3000).astype(numpy.float32)
arr_dev = Array.from_host(queue, arr)

trf = get_complex_trf(arr_dev)


# Create the FFT computation and attach the transformation above to its input.
fft = FFT(trf.parameter.output)  # (A shortcut: using the array type saved in the transformation)
fft.parameter.input.connect(trf, trf.parameter.output, new_input=trf.parameter.input)
cfft = fft.compile(queue.device)


# Run the computation
res_dev = Array.empty(queue.device, arr.shape, numpy.complex64)
cfft(queue, res_dev, arr_dev)
result = res_dev.get(queue)

reference = numpy.fft.fft(arr)

assert numpy.linalg.norm(result - reference) / numpy.linalg.norm(reference) < 1e-6  # noqa: S101, PLR2004
