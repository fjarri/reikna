import numpy

from tigger.helpers import *
from tigger.core import *


class Elementwise(Computation):

    def _get_default_basis(self):
        basis = dict(
            size=1,
            argtypes=dict(),
            code=dict(functions="", kernel=""))

        return basis

    def _get_argvalues(self, argnames, basis):
        outputs, inputs, params = argnames
        values = {name:ArrayValue((basis.size,), basis.argtypes[name])
            for name in outputs + inputs}
        values.update({name:ScalarValue(None, basis.argtypes[name])
            for name in params})

        return values

    def _get_basis_for(self, argnames, *args, **kwds):

        # Python 2 does not support explicit kwds after *args
        code = kwds.pop('code', dict(functions="", kernel=""))

        # map argument names to values
        outputs, inputs, params = argnames
        argtypes = {name:arg.dtype for name, arg in zip(outputs + inputs + params, args)}

        return dict(size=args[0].size, argtypes=argtypes, code=code)

    def _construct_operations(self, operations, argnames, basis, device_params):

        names = sum(argnames, tuple())
        template = template_from("""
        <%def name='elementwise(""" + ", ".join(names) + """)'>""" +
        basis.code.pop('functions', '') +
        """
        ${kernel_definition}
        {
            VIRTUAL_SKIP_THREADS;
            int idx = virtual_global_flat_id();
        """ +
        basis.code['kernel'] +
        """
        }
        </%def>
        """)

        operations.add_kernel(template, 'elementwise', names,
            global_size=(basis.size,), render_kwds=dict(size=basis.size))
