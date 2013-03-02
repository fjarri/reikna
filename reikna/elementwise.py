import numpy

from reikna.helpers import *
from reikna.core import *


EMPTY = dict(functions="", kernel="")


class Elementwise(Computation):
    """
    A general class for elementwise computations.

    .. py:method:: set_argnames(outputs, inputs, scalars)

        Set argument names for the computation.
        This method should be called first after the creation of the
        :py:class:`~reikna.elementwise.Elementwise` object.
        Returns ``self``.

    .. py:method:: prepare_for(*args, code=EMPTY)

        :param args: arrays and scalars, according to the lists passed to :py:meth:`set_argnames`.
        :param code: kernel code.
    """

    # For now I cannot think of any other computation requiring variable number of arguments.
    # So instead of being generic and passing argument names list to every overloaded method
    # of every computation, I employ this semi-hack with the usage of some
    # undocumented machinery from the base class.
    # Namely, Computation constructor notices missing _get_argnames() method
    # and relays initialization until _set_argnames() is called
    #
    # If other computations like this appear, I will have to return to the "generic" solution,
    # where set_argnames() method is available to any Computation object which
    # did not overload _get_argnames(), and "argnames" is passed to all overloadable methods.
    # Or, perhaps, make this method documented.
    def set_argnames(self, outputs, inputs, scalars):
        return self._set_argnames(outputs, inputs, scalars)

    def _get_argvalues(self, basis):
        outputs, inputs, params = self._get_argnames()
        values = {name:ArrayValue((basis.size,), basis.argtypes[name])
            for name in outputs + inputs}
        values.update({name:ScalarValue(basis.argtypes[name])
            for name in params})

        return values

    def _get_basis_for(self, *args, **kwds):

        # Python 2 does not support explicit kwds after *args
        code = kwds.get('code', EMPTY)

        # map argument names to values
        outputs, inputs, params = self._get_argnames()
        argtypes = {name:arg.dtype for name, arg in zip(outputs + inputs + params, args)}

        return dict(size=args[0].size, argtypes=argtypes, code=code)

    def _construct_operations(self, basis, device_params):

        operations = self._get_operation_recorder()
        names = sum(self._get_argnames(), tuple())
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
        return operations


def specialize_elementwise(outputs, inputs, scalars, code):
    """
    Returns an Elementwise class specialized for given argument names and code.

    :param outputs: a string or a list of strings with output argument names.
    :param inputs: a string or a list of strings with input argument names.
    :param scalars: ``None``, a string, or a list of strings with scalar argument names.
    :param code: ``dict(kernel, functions)`` with kernel code.
    """

    outputs = wrap_in_tuple(outputs)
    inputs = wrap_in_tuple(inputs)
    scalars = wrap_in_tuple(scalars)

    argnames = outputs + inputs + scalars

    class SpecializedElementwise(Elementwise):

        def _get_argnames(self):
            return outputs, inputs, scalars

        def _get_basis_for(self, *args):
            if len(args) != len(argnames):
                raise TypeError("The computation takes exactly " +
                    str(len(argnames)) + "arguments")
            return Elementwise._get_basis_for(self, *args, code=code)

    return SpecializedElementwise
