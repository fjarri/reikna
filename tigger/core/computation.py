import numpy
import os, os.path

from tigger.cluda.kernel import render_prelude, render_template
from tigger.cluda.dtypes import ctype, cast
import tigger.cluda.dtypes as dtypes
from tigger.core.transformation import *
from tigger.core.operation import OperationRecorder


class NotPreparedError(Exception):
    pass


class Computation:

    def __init__(self, ctx, debug=False):
        self._ctx = ctx
        self._debug = debug

        # Initialize root nodes of the transformation tree
        self._basis = AttrDict(self._get_default_basis())
        self._tr_tree = TransformationTree(*self._get_base_names())
        self._operations_prepared = False
        self._transformations_prepared = False

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
        parts = self._get_base_signature(self._basis)
        return tuple([name for name, _ in part] for part in parts)

    def _get_base_values(self):
        """
        Returns a dictionary with names and corresponding value objects for
        base computation parameters.
        """
        result = {}
        for part in self._get_base_signature(self._basis):
            result.update({name:value for name, value in part})
        return result

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
        return self._construct_basis(*self._tr_tree.base_values(), **kwds)

    def _prepare_operations(self):
        self._operations = OperationRecorder(self._ctx, self._basis, self._get_base_values())
        self._construct_operations(self._basis, self._ctx.device_params, self._operations)

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
        if new_scalar_args is None:
            new_scalar_args = []
        self._tr_tree.connect(tr, array_arg, new_array_args, new_scalar_args)
        self._transformations_prepared = False

    def set_basis(self, **kwds):
        unknown_keys = set(kwds).difference(set(self._basis))
        if len(unknown_keys) > 0:
            raise KeyError("Unknown basis keys: " + ", ".join(unknown_keys))

        if self._basis_needs_update(kwds):
            self._basis.update(kwds)
            self._operations_prepared = False

        return self

    def set_basis_for(self, *args, **kwds):
        new_basis = self._basis_for(args, kwds)
        return self.set_basis(**new_basis)

    def prepare(self, **kwds):
        """
        Prepares the computation for given basis.
        """
        self.set_basis(**kwds)

        if not self._operations_prepared:
            self._prepare_operations()
            self._operations_prepared = True
            self._transformations_prepared = False

        if not self._transformations_prepared:
            self._prepare_transformations()
            self._transformations_prepared = True
            self._operations.optimize_execution()

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
        if not self._operations_prepared or not self._transformations_prepared:
            raise NotPreparedError("The computation must be prepared before execution")

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
