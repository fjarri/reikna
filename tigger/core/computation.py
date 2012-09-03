import numpy
import os, os.path

from tigger.cluda.kernel import render_prelude, render_template
from tigger.cluda.dtypes import ctype, cast
import tigger.cluda.dtypes as dtypes
from tigger.core.transformation import *
from tigger.core.operation import OperationRecorder


class InvalidStateError(Exception):
    pass


STATE_UNDEFINED = 0
STATE_ARGNAMES_SET = 1
STATE_OPERATIONS_BUILT = 2
STATE_PREPARED = 3


class Computation:

    def __init__(self, ctx, debug=False):
        self._ctx = ctx
        self._debug = debug

        self._state = STATE_UNDEFINED

        if hasattr(self, '_get_argnames'):
        # Initialize root nodes of the transformation tree
            self._argnames = self._get_argnames()
            self._finish_init()
        else:
        # make set_argnames() visible
            self.set_argnames = self._set_argnames

    def _finish_init(self):
        self._basis = AttrDict(self._get_default_basis(self._argnames))
        self._tr_tree = TransformationTree(*self._get_base_names())
        self._state = STATE_ARGNAMES_SET

    def _set_argnames(self, outputs, inputs, scalars):
        assert self._state == STATE_UNDEFINED
        self._argnames = (tuple(outputs), tuple(inputs), tuple(scalars))
        self._finish_init()
        return self

    def get_nested_computation(self, cls):
        """
        Returns an object of supplied computation class,
        created with the same parameters (context, debug mode etc) as the current one.
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
        return self._get_argvalues(self._argnames, self._basis)

    def _get_base_dtypes(self):
        """
        Returns a dictionary with names and corresponding dtypes for
        base computation parameters.
        """
        return {name:value.dtype for name, value in self._get_base_values().items()}

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
                " arguments (" + len(args) + " given")

        # We do not need our args per se, just their properies (types and shapes).
        # So we are creating mock values to propagate through transformation tree.
        values = {}
        for i, pair_arg in enumerate(zip(pairs, args)):
            pair, arg = pair_arg
            name, value = pair
            if arg is None:
                new_value = ArrayValue(None, None) if value.is_array else ScalarValue(None, None)
            else:
                new_value = wrap_value(arg)
                if new_value.is_array != value.is_array:
                    raise TypeError("Incorrect type of argument " + str(i + 1))

            values[name] = new_value

        self._tr_tree.propagate_to_base(values)
        return self._get_basis_for(self._argnames, *self._tr_tree.base_values(), **kwds)

    def _prepare_operations(self):
        self._operations = OperationRecorder(self._ctx, self._basis, self._get_base_values())
        self._construct_operations(
            self._operations, self._argnames, self._basis, self._ctx.device_params)

    def _prepare_transformations(self):
        self._tr_tree.propagate_to_leaves(self._get_base_values())

        self._tr_tree.set_temp_nodes(self._operations.get_allocation_values())

        self._operations.prepare(self._tr_tree)
        self._leaf_signature = self.leaf_signature()

    def leaf_signature(self):
        return self._tr_tree.leaf_signature()

    def connect(self, tr, array_arg, new_array_args, new_scalar_args=None):
        """
        Connects given transformation to the external array argument.
        """
        if self._state == STATE_UNDEFINED:
            raise InvalidStateError("Base argument names are undefined")

        if new_scalar_args is None:
            new_scalar_args = []
        self._tr_tree.connect(tr, array_arg, new_array_args, new_scalar_args)
        self._state = min(self._state, STATE_OPERATIONS_BUILT)

    def set_basis(self, **kwds):
        if self._state == STATE_UNDEFINED:
            raise InvalidStateError("Base argument names are undefined")

        unknown_keys = set(kwds).difference(set(self._basis))
        if len(unknown_keys) > 0:
            raise KeyError("Unknown basis keys: " + ", ".join(unknown_keys))

        if self._basis_needs_update(kwds):
            self._basis.update(kwds)
            self._state = STATE_ARGNAMES_SET

        return self

    def set_basis_for(self, *args, **kwds):
        if self._state == STATE_UNDEFINED:
            raise InvalidStateError("Base argument names are undefined")

        new_basis = self._basis_for(args, kwds)
        return self.set_basis(**new_basis)

    def prepare(self, **kwds):
        """
        Prepares the computation for given basis.
        """
        self.set_basis(**kwds)

        if self._state == STATE_ARGNAMES_SET:
            self._prepare_operations()
            self._state = STATE_OPERATIONS_BUILT

        if self._state == STATE_OPERATIONS_BUILT:
            self._prepare_transformations()
            self._operations.optimize_execution()
            self._state = STATE_PREPARED

        return self

    def prepare_for(self, *args, **kwds):
        """
        Prepares the computation for given arguments.
        """
        new_basis = self._basis_for(args, kwds)
        return self.prepare(**new_basis)

    def signature_str(self):
        """
        Returns pretty-printed computation signature.
        This is primarily a debug method.
        """
        res = []
        for name, value in self._tr_tree.leaf_signature():
            res.append("({argtype}) {name}".format(
                name=name, argtype=str(value)))
        return ", ".join(res)

    def __call__(self, *args, **kwds):
        """
        Executes computation.
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
                " arguments (" + len(args) + " given")

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
