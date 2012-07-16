import numpy
import os, os.path

from tigger.cluda.kernel import render_prelude, render_kernel
from tigger.cluda.dtypes import ctype, cast
import tigger.cluda.dtypes as dtypes
from tigger.core.transformation import *


class Computation:

    def __init__(self, env, debug=False):
        self._env = env
        self._debug = debug

        self._basis = self._get_default_basis()
        self._tr_tree = TransformationTree(*self._get_base_names())

    def _get_base_names(self):
        parts = self._get_base_signature()
        return tuple([name for name, _ in part] for part in parts)

    def _get_base_values(self):
        result = {}
        for part in self._get_base_signature():
            result.update({name:value for name, value in part})
        return result

    def _render(self, template, **kwds):

        class PrefixHandler:

            def __init__(self, func=lambda x: x):
                self.func = func

            def __getattr__(self, name):
                return self.func(name)

        dtypes_dict = AttrDict(self._get_base_values())
        ctypes_dict = AttrDict({name:ctype(dtype) for name, dtype in dtypes_dict.items()})

        render_kwds = dict(
            dtypes=dtypes,
            basis=self._basis,
            load=PrefixHandler(load_macro_call),
            store=PrefixHandler(store_macro_call),
            param=PrefixHandler(),
            ctype=ctypes_dict,
            dtype=dtypes_dict,
            signature=signature_macro_name())

        # check that user keywords do not overlap with our keywords
        assert set(render_kwds).isdisjoint(set(kwds))

        render_kwds.update(kwds)
        return render_kernel(template, **render_kwds)

    def _basis_needs_update(self, new_basis):
        for key in new_basis:
            assert key in self._basis, "Unknown key: " + key
            if self._basis[key] != new_basis[key]:
                return True

        return False

    def _basis_for(self, args):
        pairs = self._tr_tree.leaf_signature()
        assert len(args) == len(pairs)

        values = {}
        for pair, arg in zip(pairs, args):
            name, value = pair
            new_value = wrap_value(arg)
            assert new_value.is_array == value.is_array
            values[name] = new_value

        self._tr_tree.propagate_to_base(values)
        return self._construct_basis(*self._tr_tree.base_values())

    def _change_basis(self, new_basis):
        self._basis.update(new_basis)
        self._operations = self._construct_kernels()
        self._tr_tree.propagate_to_leaves(self._get_base_values())
        for operation in self._operations:
            operation.set_env(self._env)
        self._construct_transformations()

    def _construct_transformations(self):

        for operation in self._operations:
            if not isinstance(operation, KernelCall):
                continue

            transformation_code = self._tr_tree.transformations_for(operation.argnames)
            operation.set_transformations(transformation_code)
            operation.prepare()

    def connect(self, tr, endpoint, new_endpoints, new_scalar_endpoints=None):
        if new_scalar_endpoints is None: new_scalar_endpoints = []

        if self._tr_tree.has_array_leaf(endpoint):
            self._tr_tree.connect(tr, endpoint, new_endpoints, new_scalar_endpoints)
        else:
            raise Exception("Endpoint " + endpoint + " was not found")

    def prepare(self, **kwds):
        if self._basis_needs_update(kwds):
            self._change_basis(kwds)
        return self

    def prepare_for(self, *args):
        new_basis = self._basis_for(args)
        return self.prepare(**new_basis)

    @property
    def signature(self):
        return self._tr_tree.leaf_signature()

    def __call__(self, *args):
        if self._debug:
            new_basis = self._basis_for(args)
            if self._basis_needs_update(new_basis):
                raise Exception("Given arguments require different basis")

        signature = self._tr_tree.leaf_signature()
        arg_dict = {}
        for pair, arg in zip(signature, args):
            name, value = pair
            if isinstance(value, ScalarValue):
                arg = cast(value.dtype)(arg)
            arg_dict[name] = arg

        for operation in self._operations:
            operation(*args)


class KernelCall:

    def __init__(self, name, argnames, base_src, block=(1,1,1), grid=(1,1), shared=0):
        self.name = name
        self.argnames = argnames
        self.block = block
        self.grid = grid
        self.src = base_src
        self.shared = shared

    def set_env(self, env):
        self.env = env
        self.prelude = render_prelude(env)

    def set_transformations(self, tr_code):
        self.tr_code = tr_code

    def prepare(self):
        self.full_src = self.prelude + self.tr_code + self.src
        print self.full_src
        self.module = self.env.compile(self.full_src)
        self.kernel = getattr(self.module, self.name)

    def __call__(self, *args):
        self.kernel(*args, block=self.block, grid=self.grid, shared=self.shared)
