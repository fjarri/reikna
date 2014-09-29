"""
This example illustrates how to:
- define a custom structure type with a correct alignment
  (allowing it to be used to exchange data between numpy and kernels);
- construct a reduction computation that operates on arrays of the struct type above.
"""

import numpy
from reikna.cluda import dtypes, any_api
from reikna.algorithms import Reduce, Predicate
from reikna.cluda import Snippet
from reikna.core import Annotation, Type, Transformation, Parameter


# Pick the first available GPGPU API and make a Thread on it.
api = any_api()
thr = api.Thread.create()


# Minmax data type and the corresponding structure.
# Note that the dtype must be aligned before use on a GPU.
mmc_dtype = dtypes.align(numpy.dtype([
    ("cur_min", numpy.int32),
    ("cur_max", numpy.int32),
    ("pad", numpy.int32),
    ]))
mmc_c_decl = dtypes.ctype_module(mmc_dtype)


# Create the "empty" element for our minmax monoid, that is
# x `minmax` empty == empty `minmax` x == x.
empty = numpy.empty(1, mmc_dtype)[0]
empty["cur_min"] = 1 << 30
empty["cur_max"] = -(1 << 30)


# Reduction predicate for the minmax.
# v1 and v2 get the names of two variables to be processed.
predicate = Predicate(
    Snippet.create(lambda v1, v2: """
        ${ctype} result = ${v1};
        if (${v2}.cur_min < result.cur_min)
            result.cur_min = ${v2}.cur_min;
        if (${v2}.cur_max > result.cur_max)
            result.cur_max = ${v2}.cur_max;
        return result;
        """,
        render_kwds=dict(ctype=mmc_c_decl)),
    empty)


# Test array
arr = numpy.random.randint(0, 10**6, 20000)


# A transformation that creates initial minmax structures for the given array of integers
to_mmc = Transformation(
    [Parameter('output', Annotation(Type(mmc_dtype, arr.shape), 'o')),
    Parameter('input', Annotation(arr, 'i'))],
    """
    ${output.ctype} res;
    res.cur_min = ${input.load_same};
    res.cur_max = ${input.load_same};
    ${output.store_same}(res);
    """)


# Create the reduction computation and attach the transformation above to its input.
reduction = Reduce(to_mmc.output, predicate)
reduction.parameter.input.connect(to_mmc, to_mmc.output, new_input=to_mmc.input)
creduction = reduction.compile(thr)


# Run the computation
arr_dev = thr.to_device(arr)
res_dev = thr.empty_like(reduction.parameter.output)
creduction(res_dev, arr_dev)
minmax = res_dev.get()

assert minmax["cur_min"] == arr.min()
assert minmax["cur_max"] == arr.max()
