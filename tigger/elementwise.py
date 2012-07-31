import numpy
from tigger.core import *


class Elementwise(Computation):

    def __init__(self, ctx, code, signature, **kwds):
        self._code = code
        self._base_stores, self._base_loads, self._base_params = signature
        Computation.__init__(self, ctx, **kwds)

    def _get_default_basis(self):
        res = {(name + '_dtype'):numpy.float32 for name in
            self._base_stores + self._base_loads + self._base_params}
        res['size'] = 1
        return res

    def _construct_basis(self, *args):

        bs = dict(size=args[0].size)

        names = self._base_stores + self._base_loads + self._base_params
        for arg, name in zip(args, names):
            bs[name + '_dtype'] = arg.dtype

        return bs

    def _get_base_signature(self):
        bs = self._basis
        stores = [(name, ArrayValue(None, bs[name + '_dtype'])) for name in self._base_stores]
        stores[0][1].shape = (bs.size,)
        loads = [(name, ArrayValue(None, bs[name + '_dtype'])) for name in self._base_loads]
        params = [(name, ScalarValue(None, bs[name + '_dtype'])) for name in self._base_params]

        return stores, loads, params

    def _construct_kernels(self):
        bs = self._basis

        template = template_from("""
        KERNEL void elementwise(${signature})
        {
            int idx = ID_FLAT;
            int size = ${size};
            if (idx < ${size})
            {
        """ +
        self._code +
        """
            }
        }
        """)

        src = self._render(template, size=bs.size)

        names = self._base_stores + self._base_loads + self._base_params
        return [KernelCall(
            'elementwise', names, src,
            global_size=bs.size
        )]
