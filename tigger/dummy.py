import numpy

from tigger.core.computation import *
from tigger.core.transformation import *
from tigger.core.helpers import AttrDict

TEMPLATE = loadTemplateFor(__file__)


class Dummy(Computation):

    def _get_default_basis(self):
        return AttrDict(a_dtype=numpy.float32, b_dtype=numpy.float32, c_dtype=numpy.float32,
            coeff_dtype=numpy.float32)

    def _construct_basis(self, C, A, B, coeff):
        bs = AttrDict()
        bs.a_dtype = A.dtype
        bs.b_dtype = B.dtype
        bs.c_dtype = C.dtype
        bs.coeff_dtype = coeff.dtype
        bs.size = C.size
        return bs

    def _get_base_signature(self):
        bs = self._basis
        shape = (bs.size,)
        return [('C', MockValue.array(shape, bs.c_dtype))], \
            [
                ('A', MockValue.array(shape, bs.a_dtype),
                ('B', MockValue.array(shape, bs.b_dtype)], \
            [('coeff', MockValue.scalar(None, bs.c_dtype)]

    def _construct_derived(self):
        bp = self._basis

        block_size = self._env.params.max_block_size
        module = self._env.compile(TEMPLATE, bp=bp, dp=dp)

        self._set_kernels([KernelCall(
            grid=(block_size, 1),
            block=(bp.size / block_size, 1, 1),
            kernel=module.dummy
        )])

        return [
            # list of kernels here
        ]
