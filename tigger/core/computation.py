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
        self._prepared = False

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

        dtypes_dict = AttrDict(self._get_base_dtypes())
        ctypes_dict = AttrDict({name:ctype(dtype) for name, dtype in dtypes_dict.items()})

        store_names, load_names, param_names = self._get_base_names()
        load_dict = AttrDict({name:load_macro_call(name) for name in load_names})
        store_dict = AttrDict({name:store_macro_call(name) for name in store_names})
        param_dict = AttrDict({name:leaf_name(name) for name in param_names})

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
            if key not in self._basis:
                raise KeyError("Unknown basis key: " + key)
            if self._basis[key] != new_basis[key]:
                return True

        return False

    def _basis_for(self, args):
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
        return self._construct_basis(*self._tr_tree.base_values())

    def _change_basis(self, new_basis):
        """
        Performs all necessary operations corresponding to the change of basis.
        Updates basis, recreates operations, updates transformation tree.
        """
        self._basis.update(new_basis)
        self._operations = self._construct_kernels()
        self._tr_tree.propagate_to_leaves(self._get_base_values())
        for operation in self._operations:
            operation.prepare(self._ctx, self._tr_tree)
        self._leaf_signature = self._tr_tree.leaf_signature()

    def connect(self, tr, array_arg, new_array_args, new_scalar_args=None):
        """
        Connects given transformation to the external array argument.
        """
        if new_scalar_args is None:
            new_scalar_args = []
        self._tr_tree.connect(tr, array_arg, new_array_args, new_scalar_args)
        self._prepared = False

    def prepare(self, **kwds):
        """
        Prepares the computation for given basis.
        """
        if self._basis_needs_update(kwds) or not self._prepared:
            self._change_basis(kwds)
            self._prepared = True
        return self

    def prepare_for(self, *args):
        """
        Prepares the computation for given arguments.
        """
        new_basis = self._basis_for(args)
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

    def __call__(self, *args):
        """
        Executes computation.
        """
        if not self._prepared:
            raise NotPreparedError("The computation must be prepared before execution")

        if self._debug:
            new_basis = self._basis_for(args)
            if self._basis_needs_update(new_basis):
                raise ValueError("Given arguments require different basis")

        if len(args) != len(self._leaf_signature):
            raise TypeError("Computation takes " + str(len(self._leaf_signature)) +
                " arguments (" + len(args) + " given")

        # Assign arguments to names and cast scalar values
        arg_dict = {}
        for pair, arg in zip(self._leaf_signature, args):
            name, value = pair
            if not value.is_array:
                arg = cast(value.dtype)(arg)
            arg_dict[name] = arg

        # Call kernels with argument list based on their base arguments
        for operation in self._operations:
            op_args = [arg_dict[name] for name in operation.leaf_argnames]
            operation(*args)


class KernelCall:

    def __init__(self, name, base_argnames, base_src, block=(1,1,1), grid=(1,1), shared=0):
        self.name = name
        self.base_argnames = base_argnames
        self.block = block
        self.grid = grid
        self.src = base_src
        self.shared = shared

    def prepare(self, ctx, tr_tree):
        transformation_code = tr_tree.transformations_for(self.base_argnames)
        self.full_src = transformation_code + self.src
        self.module = ctx.compile(self.full_src)
        self.kernel = getattr(self.module, self.name)
        self.leaf_argnames = [name for name, _ in tr_tree.leaf_signature(self.base_argnames)]

    def __call__(self, *args):
        self.kernel(*args, block=self.block, grid=self.grid, shared=self.shared)
