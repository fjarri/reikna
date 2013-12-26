import numpy
import pytest

from helpers import *
from reikna.linalg import EntrywiseNorm


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('order', [0.5, 1, 2])
def test_entrywise_norm(thr, dtype, order):

    a = get_test_array(1000, dtype)
    a_dev = thr.to_device(a)

    norm = EntrywiseNorm(a_dev, order=order)

    b_dev = thr.empty_like(norm.parameter.output)
    b_ref = numpy.linalg.norm(a, ord=order).astype(norm.parameter.output.dtype)

    normc = norm.compile(thr)
    normc(b_dev, a_dev)

    assert diff_is_negligible(b_dev.get(), b_ref)
