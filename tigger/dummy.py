import numpy

from tigger.core.computation import *
from tigger.core.transformation import *
from tigger.core.helpers import AttrDict, template_for

_DUMMY_TEMPLATE = template_for(__file__)


class Dummy(Computation):

    def _get_default_basis(self):
        return AttrDict(arr_dtype=numpy.float32, coeff_dtype=numpy.float32, size=1)

    def _construct_basis(self, C, A, B, coeff):
        bs = AttrDict()
        assert C.dtype == A.dtype and C.dtype == B.dtype
        bs.arr_dtype = C.dtype
        bs.coeff_dtype = coeff.dtype
        bs.size = C.size
        return bs

    def _get_base_signature(self):
        bs = self._basis
        shape = (bs.size,)
        return [('C', ArrayValue(shape, bs.arr_dtype))], \
            [
                ('A', ArrayValue(shape, bs.arr_dtype)),
                ('B', ArrayValue(shape, bs.arr_dtype))], \
            [('coeff', ScalarValue(None, bs.coeff_dtype))]

    def _construct_kernels(self):
        # basis will be passed automatically as a keyword
        # optional keywords can be passed here
        # TODO: is it a good way to specify templates?
        bs = self._basis
        src = self._render(_DUMMY_TEMPLATE, size=bs.size)
        block_size = 128

        return [KernelCall(
            'dummy', # name of function to call from the rendered template
            # possible shortcut - if identical to global signature, just pass None
            ['C', 'A', 'B', 'coeff'],
            src, # actual source with necessary placeholders (SIGNATURE, LOAD_... etc)
            grid=(block_size, 1),
            block=(bs.size / block_size, 1, 1)
        )]
