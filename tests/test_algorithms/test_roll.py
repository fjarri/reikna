import itertools

import numpy
import pytest
from grunnur import Array, Snippet, dtypes

from helpers import diff_is_negligible, get_test_array
from reikna import helpers
from reikna.algorithms import Roll, RollInplace

shapes = [
    (10,),
    (10, 20),
    (10, 20, 30),
]
shapes_and_axes = [
    (shape, axis)
    for shape, axis in itertools.product(shapes, [-1, 0, 1, 2])
    if axis is None or axis < len(shape)
]
shapes_and_axes_ids = [str(shape) + "," + str(axis) for shape, axis in shapes_and_axes]


@pytest.mark.parametrize(("shape", "axis"), shapes_and_axes, ids=shapes_and_axes_ids)
def test_roll(queue, shape, axis):
    dtype = numpy.int32
    shift = 4

    arr = get_test_array(shape, dtype)
    arr_dev = Array.from_host(queue, arr)

    roll = Roll(arr_dev, axis=axis).compile(queue.device)
    res_dev = Array.empty_like(queue.device, roll.parameter.output)

    roll(queue, res_dev, arr_dev, shift)
    queue.synchronize()

    res_test = res_dev.get(queue)
    res_ref = numpy.roll(arr, shift, axis=axis)

    assert diff_is_negligible(res_ref, res_test)


@pytest.mark.parametrize(("shape", "axis"), shapes_and_axes, ids=shapes_and_axes_ids)
def test_roll_inplace(queue, shape, axis):
    dtype = numpy.int32
    shift = 4

    arr = get_test_array(shape, dtype)
    arr_dev = Array.from_host(queue, arr)

    roll = RollInplace(arr_dev, axis=axis).compile(queue.device)

    roll(queue, arr_dev, shift)
    queue.synchronize()

    res_test = arr_dev.get(queue)
    res_ref = numpy.roll(arr, shift, axis=axis)

    assert diff_is_negligible(res_ref, res_test)
