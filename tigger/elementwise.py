import numpy

from tigger.helpers import *
from tigger.core import *


EMPTY = dict(functions="", kernel="")


class Elementwise(Computation):
    """
    A general class for elementwise computations.

    .. py:method:: set_argnames(outputs, inputs, scalars)

    .. py:method:: prepare_for(*args, code=EMPTY)

        :param args: arrays and scalars, according to the lists passed to :py:meth:`set_argnames`.
        :param code: kernel code.

    .. py:method:: prepare(size=1, argtypes={}, code=EMPTY)

        :param size: base size for the computation (== number of work items to use)
        :param argtypes: dictionary containing dtypes associated with argument names
            passed to :py:meth:`set_argnames`.
        :param code: kernel code.
    """

    def _get_default_basis(self):
        basis = dict(
            size=1,
            argtypes=dict(),
            code=EMPTY)

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
        code = kwds.pop('code', EMPTY)

        # map argument names to values
        outputs, inputs, params = argnames
        argtypes = {name:arg.dtype for name, arg in zip(outputs + inputs + params, args)}

        return dict(size=args[0].size, argtypes=argtypes, code=code)

    def _construct_operations(self, operations, argnames, basis, device_params):

        names = sum(argnames, tuple())
        name_str = ", ".join(names)

        template = template_from(
            template_defs_for_code(basis.code, names) +
            """
            <%def name='elementwise(""" + name_str  + """)'>
            ${code_functions(""" + name_str + """)}
            ${kernel_definition}
            {
                VIRTUAL_SKIP_THREADS;
                int idx = virtual_global_flat_id();
                ${code_kernel(""" + name_str + """)}
            }
            </%def>
            """)

        operations.add_kernel(template, 'elementwise', names,
            global_size=(basis.size,), render_kwds=dict(size=basis.size))
