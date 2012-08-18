import numpy
import os, os.path

from tigger.cluda.kernel import render_prelude, render_template
from tigger.cluda.dtypes import ctype, cast
import tigger.cluda.dtypes as dtypes
from tigger.core.transformation import *


class NotPreparedError(Exception):
    pass


class Computation:

    def __init__(self, ctx, debug=False):
        self._ctx = ctx
        self._debug = debug

        # Initialize root nodes of the transformation tree
        self._basis = AttrDict(self._get_default_basis())
        self._tr_tree = TransformationTree(*self._get_base_names())
        self._operations = []
        self._operations_prepared = False
        self._transformations_prepared = False

    def _get_base_names(self):
        """
        Returns three lists (outs, ins, scalars) with names of base computation parameters.
        """
        parts = self._get_base_signature()
        return tuple([name for name, _ in part] for part in parts)

    def _get_base_values(self):
        """
        Returns a dictionary with names and corresponding value objects for
        base computation parameters.
        """
        result = {}
        for part in self._get_base_signature():
            result.update({name:value for name, value in part})
        return result

    def _get_base_dtypes(self):
        """
        Returns a dictionary with names and corresponding dtypes for
        base computation parameters.
        """
        return {name:value.dtype for name, value in self._get_base_values().items()}

    def _render(self, template, **kwds):
        """
        Renders given template of the computation kernel.
        Called from derived class.
        """

        internal_inputs = kwds.pop('internal_inputs', [])
        internal_outputs = kwds.pop('internal_outputs', [])

        dtypes_dict = AttrDict(self._get_base_dtypes())
        ctypes_dict = AttrDict({name:ctype(dtype) for name, dtype in dtypes_dict.items()})

        store_names, load_names, param_names = self._get_base_names()
        load_dict = AttrDict({name:load_macro_call(name)
            for name in load_names + internal_inputs})
        store_dict = AttrDict({name:store_macro_call(name)
            for name in store_names + internal_outputs})
        param_dict = AttrDict({name:leaf_name(name)
            for name in param_names})

        render_kwds = dict(
            basis=self._basis,
            load=load_dict,
            store=store_dict,
            param=param_dict,
            ctype=ctypes_dict,
            dtype=dtypes_dict,
            signature=signature_macro_name())

        # check that user keywords do not overlap with our keywords
        intersection = set(render_kwds).intersection(kwds)
        if len(intersection) > 0:
            raise ValueError("Render keywords clash with internal variables: " +
                ", ".join(intersection))

        render_kwds.update(kwds)
        return render_template(template, **render_kwds)

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

    def _change_basis(self, new_basis):
        """
        Performs all necessary operations corresponding to the change of basis.
        Updates basis, recreates operations, updates transformation tree.
        """
        self._basis.update(new_basis)
        self._operations = self._construct_kernels()

    def _prepare_transformations(self):
        self._tr_tree.propagate_to_leaves(self._get_base_values())

        self._tr_tree.clear_temp_nodes()
        for operation in self._operations:
            if isinstance(operation, Allocate):
                self._tr_tree.add_temp_node(operation.name, ArrayValue(None, operation.dtype))
            if isinstance(operation, KernelCall):
                operation.prepare(self._ctx, self._tr_tree)
            elif isinstance(operation, ComputationCall):
                operation.prepare()

        self._leaf_signature = self._tr_tree.leaf_signature()

    def _optimize_execution(self):

        # In theory, we can optimize the usage of temporary buffers with help of views
        # Now we just allocate them separately
        self._temp_buffers = {}
        allocations = [op for op in self._operations if isinstance(op, Allocate)]
        for allocation in allocations:
            self._temp_buffers[allocation.name] = self._ctx.allocate(
                allocation.shape, allocation.dtype)

        self._calls = [op for op in self._operations if not isinstance(op, Allocate)]

    def connect(self, tr, array_arg, new_array_args, new_scalar_args=None):
        """
        Connects given transformation to the external array argument.
        """
        if new_scalar_args is None:
            new_scalar_args = []
        self._tr_tree.connect(tr, array_arg, new_array_args, new_scalar_args)
        for op in self._operations:
            if isinstance(op, ComputationCall):
                op.connect(
                    tr, array_arg, new_array_args, new_scalar_args=new_scalar_args)

        self._transformations_prepared = False

    def prepare(self, **kwds):
        """
        Prepares the computation for given basis.
        """
        unknown_keys = set(kwds).difference(set(self._basis))
        if len(unknown_keys) > 0:
            raise KeyError("Unknown basis keys: " + ", ".join(unknown_keys))

        if not self._operations_prepared or self._basis_needs_update(kwds):
            self._change_basis(kwds)
            self._operations_prepared = True
            self._transformations_prepared = False
        if not self._transformations_prepared:
            self._prepare_transformations()
            self._transformations_prepared = True
            self._optimize_execution()
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
        arg_dict = dict(self._temp_buffers)
        for pair, arg in zip(self._leaf_signature, args):
            name, value = pair
            if not value.is_array:
                arg = cast(value.dtype)(arg)
            assert name not in arg_dict
            arg_dict[name] = arg

        # Call kernels with argument list based on their base arguments
        for operation in self._calls:
            op_args = [arg_dict[name] for name in operation.leaf_argnames]
            operation(*op_args)


class Allocate:

    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class ComputationCall:

    def __init__(self, computation, *argnames):
        self.computation = computation
        self.argnames = argnames

        argnames = [x for x, _ in self.computation._leaf_signature]
        self.map_to_internal = {external_name:internal_name
            for external_name, internal_name in zip(self.argnames, argnames)}
        self.map_to_external = {internal_name:external_name
            for external_name, internal_name in zip(self.argnames, argnames)}

    def prepare(self):
        self.computation.prepare()
        replace = lambda x: self.map_to_external.get(x, x)
        argnames = [x for x, _ in self.computation._leaf_signature]
        self.leaf_argnames = [replace(name) for name in argnames]

    def __call__(self, *args):
        self.computation(*args)

    def connect(self, tr, array_arg, new_array_args, new_scalar_args=None):
        replace = lambda x: self.map_to_internal.get(x, x)
        array_arg = replace(array_arg)
        new_array_args = [replace(name) for name in new_array_args]
        new_scalar_args = [replace(name) for name in new_scalar_args]

        self.computation.connect(tr, array_arg, new_array_args, new_scalar_args)


class KernelCall:

    def __init__(self, name, base_argnames, base_src, global_size,
            local_size=None, shared=0):
        self.name = name
        self.base_argnames = list(base_argnames)
        self.local_size = local_size
        self.global_size = global_size
        self.src = base_src
        self.shared = shared

    def prepare(self, ctx, tr_tree):
        transformation_code = tr_tree.transformations_for(self.base_argnames)
        self.full_src = transformation_code + self.src
        self.module = ctx.compile(self.full_src)
        self.kernel = getattr(self.module, self.name)
        self.leaf_argnames = [name for name, _ in tr_tree.leaf_signature(self.base_argnames)]
        self.kernel.prepare(self.global_size, local_size=self.local_size, shared=self.shared)

    def __call__(self, *args):
        self.kernel.prepared_call(*args)
