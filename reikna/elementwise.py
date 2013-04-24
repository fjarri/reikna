import itertools
import numpy

from reikna.cluda import Snippet
from reikna.helpers import *
from reikna.core import *


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
        :param code: a function that takes argument mocks as parameters and returns
            a snippet with the kernel body.
        :param dependencies: optional, a list of pairs of argument names
            whose arrays depend on each other.
            By default, only output arguments are dependent on each other.
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

        # map argument names to values
        outputs, inputs, params = self._get_argnames()
        argtypes = {name:arg.dtype for name, arg in zip(outputs + inputs + params, args)}

        # Python 2 does not support explicit kwds after *args
        code = kwds.get('code', None)
        if code is None:
            code = lambda *code_args: Snippet(template_def(
                outputs + inputs + params, ""))

        dependencies = kwds.get('dependencies', None)
        if dependencies is None:
            dependencies = []
        dependencies.extend([(arg1, arg2) for arg1, arg2
            in itertools.product(outputs, outputs) if arg1 != arg2])

        snippet = code(*args)

        return dict(size=args[0].size, argtypes=argtypes,
            snippet=snippet, dependencies=dependencies)

    def _construct_operations(self, basis, device_params):

        operations = self._get_operation_recorder()
        names = sum(self._get_argnames(), tuple())
        name_str = ", ".join(names)

        template = template_def(
            names,
            """
            ${kernel_definition}
            {
                VIRTUAL_SKIP_THREADS;
                int idx = virtual_global_flat_id();
                ${basis.snippet(""" + ",".join(names) + """)}
            }
            """)

        operations.add_kernel(template, names,
            global_size=(basis.size,),
            dependencies=basis.dependencies)
        return operations


def specialize_elementwise(outputs, inputs, scalars, code, dependencies=None):
    """
    Returns an Elementwise class specialized for given argument names and code.

    :param outputs: a string or a list of strings with output argument names.
    :param inputs: a string or a list of strings with input argument names.
    :param scalars: ``None``, a string, or a list of strings with scalar argument names.
    :param code: see the ``code`` argument of Elementwise's
        :py:meth:`~reikna.elementwise.Elementwise.prepare_for`.
    :param dependencies: see the ``dependencies`` argument of Elementwise's
        :py:meth:`~reikna.elementwise.Elementwise.prepare_for`.
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
            return Elementwise._get_basis_for(self, *args, code=code, dependencies=dependencies)

    return SpecializedElementwise
