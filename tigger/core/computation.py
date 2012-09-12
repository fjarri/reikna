import numpy
import os, os.path

from tigger.cluda.kernel import render_prelude, render_template
from tigger.cluda.dtypes import ctype, cast
import tigger.cluda.dtypes as dtypes
from tigger.core.transformation import *
from tigger.core.operation import OperationRecorder


class InvalidStateError(Exception):
    pass


# Computation is not ready for calling overloaded methods from derived classes.
STATE_NOT_INITIALIZED = 0
# Computation is initialized and ready for calling preparations
# or adding transformations.
STATE_INITIALIZED = 1
# Computation is fully prepared and ready to use
STATE_PREPARED = 2


class Computation:
    """
    Creates a computation class and performs basic initialization for the
    :py:class:`~tigger.cluda.api.Context` object ``ctx``.
    Note that the computation is unusable until :py:func:`prepare`
    or :py:func:`prepare_for` is called.
    If ``debug`` is ``True``, a couple of additional checks will be performed in runtime
    during preparation and calls to computation.

    The following methods are for overriding by computations
    inheriting :py:class:`Computation` class.

    .. py:method:: _set_argnames(outputs, inputs, scalars)

        Special method to use by computations with variable number of arguments.
        Should be called before any connections and preparations are made.

    .. py:method:: _get_argnames()

        Must return a tuple ``(outputs, inputs, scalars)``, where each of
        ``outputs``, ``inputs``, ``scalars`` is a tuple of argument names used by this computation.
        If this method is not overridden, :py:meth:`set_argnames` will have to be called
        right after creating the computation object.

    .. py:method:: _get_argvalues(argnames, basis)

        Must return a dictionary with :py:class:`~tigger.core.ArrayValue` and
        :py:class:`~tigger.core.ScalarValue` objects assigned to the argument names.

    .. py:method:: _get_basis_for(*args, **kwds)

        Must return a dictionary with basis values for the computation working with ``args``,
        given optional parameters ``kwds``.
        If names of positional and keyword arguments are known in advance,
        it is better to use them explicitly in the signature.

    .. py:method:: _construct_operations(operations, basis, device_params)

        Must fill the ``operations`` object with actions required to execute the computation.
        See the :py:class:`~tigger.core.operation.OperationRecorder` class reference
        for the list of available actions.

    The rest is public methods.
    """

    def __init__(self, ctx, debug=False):
        self._ctx = ctx
        self._debug = debug

        self._state = STATE_NOT_INITIALIZED

        # finish initialization only if the computation has fixed argument list
        if hasattr(self, '_get_argnames'):
            self._argnames = self._get_argnames()
            self._finish_init()

    def _finish_init(self):
        self._tr_tree = TransformationTree(*self._get_base_names())
        self._state = STATE_INITIALIZED

    def _set_argnames(self, outputs, inputs, scalars):
        if self._state != STATE_NOT_INITIALIZED:
            raise InvalidStateError("Argument names were already set once")
        self._argnames = (tuple(outputs), tuple(inputs), tuple(scalars))
        self._finish_init()
        return self

    def get_nested_computation(self, cls):
        """
        Calls ``cls`` constructor with the same arguments and keywords
        as were given to its own constructor.
        """
        return cls(self._ctx, debug=self._debug)

    def _get_base_names(self):
        """
        Returns three lists (outs, ins, scalars) with names of base computation parameters.
        """
        return self._argnames

    def _get_base_values(self):
        """
        Returns a dictionary with names and corresponding value objects for
        base computation parameters.
        """
        return self._get_argvalues(self._basis)

    def _basis_needs_update(self, new_basis):
        """
        Tells whether ``new_basis`` has some values differing from the current basis.
        """
        for key in new_basis:
            if self._basis[key] != new_basis[key]:
                return True

        return False

    def _basis_for(self, args, kwds):
        """
        Returns the basis necessary for processing given external arguments.
        """
        pairs = self._tr_tree.leaf_signature()
        if len(args) != len(pairs):
            raise TypeError("Computation takes " + str(len(pairs)) +
                " arguments (" + str(len(args)) + " given)")

        # We do not need our args per se, just their properies (types and shapes).
        # So we are creating mock values to propagate through transformation tree.
        values = {}
        for i, pair_arg in enumerate(zip(pairs, args)):
            pair, arg = pair_arg
            name, value = pair
            if arg is None:
                new_value = ArrayValue(None, None) if value.is_array else ScalarValue(None)
            else:
                new_value = wrap_value(arg)
                if new_value.is_array != value.is_array:
                    raise TypeError("Incorrect type of argument " + str(i + 1))

            values[name] = new_value

        self._tr_tree.propagate_to_base(values)
        return self._get_basis_for(*self._tr_tree.base_values(), **kwds)

    def _prepare_operations(self):
        self._operations = OperationRecorder(self._ctx, self._basis, self._get_base_values())
        self._construct_operations(
            self._operations, self._basis, self._ctx.device_params)

    def _prepare_transformations(self):
        self._tr_tree.set_temp_nodes(self._operations.get_allocation_values())

        self._operations.prepare(self._tr_tree)
        self._leaf_signature = self.leaf_signature()

    def leaf_signature(self):
        return self._tr_tree.leaf_signature()

    def connect(self, tr, array_arg, new_array_args, new_scalar_args=None):
        """
        Connects a :py:class:`~tigger.core.Transformation` instance to the computation.
        After the successful connection the computation resets to teh unprepared state.

        :param array_arg: name of the leaf computation parameter to connect to.
        :param new_array_args: list of the names for the new leaf array parameters.
        :param new_scalar_args: list of the names for the new leaf scalar parameters.
        """
        if self._state != STATE_INITIALIZED:
            raise InvalidStateError(
                "Cannot connect transformations after the computation has been prepared")

        if new_scalar_args is None:
            new_scalar_args = []
        self._tr_tree.connect(tr, array_arg, new_array_args, new_scalar_args)

    def prepare_for(self, *args, **kwds):
        """
        Prepare the computation so that it could run with ``args`` supplied to :py:meth:`__call__`.
        """
        if self._state == STATE_NOT_INITIALIZED:
            raise InvalidStateError("Computation is not fully initialized")
        elif self._state == STATE_PREPARED:
            raise InvalidStateError("Cannot prepare the same computation twice")

        self._basis = AttrDict(self._basis_for(args, kwds))
        self._prepare_operations()
        self._prepare_transformations()
        self._operations.optimize_execution()
        self._state = STATE_PREPARED

        return self

    def signature_str(self):
        """
        Returns a string with the signature of the computation,
        containing argument names, types and shapes (in case of arrays).

        This is primarily a debug method.
        """
        res = []
        for name, value in self._tr_tree.leaf_signature():
            res.append("({argtype}) {name}".format(
                name=name, argtype=str(value)))
        return ", ".join(res)

    def __call__(self, *args, **kwds):
        """
        Execute computation with given arguments.
        The order and types of arguments are defined by the base computation
        and connected transformations.
        The signature can be also viewed by means of :py:meth:`signature_str`.
        """
        if self._state != STATE_PREPARED:
            raise InvalidStateError("The computation must be fully prepared before execution")

        if self._debug:
            new_basis = self._basis_for(args, kwds)
            if self._basis_needs_update(new_basis):
                raise ValueError("Given arguments require different basis")
        else:
            if len(kwds) > 0:
                raise ValueError("Keyword arguments should be passed to prepare_for()")

        if len(args) != len(self._leaf_signature):
            raise TypeError("Computation takes " + str(len(self._leaf_signature)) +
                " arguments (" + str(len(args)) + " given)")

        # Assign arguments to names and cast scalar values
        arg_dict = dict(self._operations.allocations)
        for pair, arg in zip(self._leaf_signature, args):
            name, value = pair
            if not value.is_array:
                arg = cast(value.dtype)(arg)

            assert name not in arg_dict
            arg_dict[name] = arg

        # Call kernels with argument list based on their base arguments
        for operation in self._operations.operations:
            op_args = [arg_dict[name] for name in operation.leaf_argnames]
            operation(*op_args)
