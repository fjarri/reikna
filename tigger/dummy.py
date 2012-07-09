import numpy
import helpers
from .helpers import *
from .cluda.helpers import *


TEMPLATE = loadTemplateFor(__file__)


class Dummy(Computation):

    def _get_default_basis(self):
        return dict(a_dtype=numpy.float32, b_dtype=numpy.float32, c_dtype=numpy.float32,
            coeff_dtype=numpy.float32)

    def _construct_basis(self, args):
        bs = Computation._construct_basis(self, args)
        bs.a_dtype = args.A.dtype
        bs.b_dtype = args.B.dtype
        bs.c_dtype = args.C.dtype
        bs.coeff_dtype = args.coeff.dtype
        bs.size = args.a.size
        return bs

    def _get_base_endpoints(self):
        return ['C'], ['A', 'B'], ['coeff']

    def _construct_derived(self):
        dp = Computation._construct_derived(self)
        bp = self._basis

        block_size = self._env.params.max_block_size
        module = self._env.compile(TEMPLATE, bp=bp, dp=dp)

        self._set_kernels([KernelCall(
            grid=(block_size, 1),
            block=(bp.size / block_size, 1, 1),
            kernel=module.dummy
        )])

        return dp
