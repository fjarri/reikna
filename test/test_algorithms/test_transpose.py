import itertools

import numpy
import pytest

from helpers import *

from reikna.algorithms import Transpose


def pytest_generate_tests(metafunc):
    if 'shape_and_axes' in metafunc.funcargnames:
        dims = [2, 3, 4]
        shape_sets = {
            2: [(13, 11), (511, 134), (77, 2049)],
            3: [(13, 33, 1029), (77, 55, 33)],
            4: [(11, 13, 19, 31), (35, 4, 57, 8)]}
        size_limit = 2 ** 20

        axes = [None, (1, 0), (1, 0, 2), (2, 0, 1), (2, 3, 0, 1), (0, 3, 1, 2)]

        vals = []
        for dim in dims:
            shapes = shape_sets[dim]
            axes_vals = [a for a in axes if a is None or len(a) == dim]
            vals += itertools.product(shapes, axes_vals)

        metafunc.parametrize('shape_and_axes', vals,
            ids=[str(shape) + "," + str(axes) for shape, axes in vals])

def test_errors(thr, shape_and_axes):
    shape, axes = shape_and_axes
    a = get_test_array(shape, numpy.int32)
    a_dev = thr.to_device(a)
    res_ref = numpy.transpose(a, axes)

    tr = Transpose(a, axes=axes)
    res_dev = thr.empty_like(tr.parameter.output)

    tr = tr.compile(thr)
    tr(res_dev, a_dev)

    assert diff_is_negligible(res_dev.get(), res_ref)
