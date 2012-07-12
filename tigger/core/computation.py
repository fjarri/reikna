import numpy
import os, os.path

from tigger.cluda import render
from tigger.cluda.dtypes import ctype


class Computation:

    def __init__(self, env, debug=False):
        self._env = env
        self._debug = debug

        # DERIVED: returns {} with basis parameters
        self._basis = AttrDict(**self._get_default_basis())

        stores, loads, scalars = self._get_base_names()

        # TRANS: takes [], [], [] with endpoint names
        self._tr_tree = TransformationTree(stores, loads, scalars)

    def _get_base_names(self):
        # DERIVED: returns [(name, mock value)], [], []
        stores, loads, scalars = self._get_base_signature()
        get_names = lambda x: [name for name, _ in x]
        return get_names(stores), get_names(loads), get_names(scalars)

    def _get_base_values(self):
        get_values = lambda x: {name:value for name, value in x}

        result = {}
        # DERIVED: returns [(name, mock value)], [], []
        for x in self._get_base_signature():
            result.update(get_values(x))
        return result

    def _basis_needs_update(self, new_basis):
        for key in new_basis:
            assert key in self._basis, "Unknown key: " + key
            if self._basis[key] != new_basis[key]:
                return True

        return False

    def _basis_for(self, args):
        # TRANS: returns [name], [], [] with endpoint names
        stores, loads, scalars = self._tr_tree.leaf_signature()
        pairs = stores + loads + scalars

        assert len(args) == len(pairs)

        values = {}
        for pair, arg for zip(pairs, args):
            name, value = pair
            new_value = MockValue(arg)
            assert new_value.is_array == value.is_array
            values[name] = new_value

        # TRANS: takes {name: mock_val} and propagates it from leaves to roots,
        # returning list of MockValue instances
        new_base_args = propagate_to_base(self._tr_tree, values)

        return self._construct_basis(*new_base_args)

    def _change_basis(self, new_basis):
        self._basis.update(new_basis)
        self._operations = self._construct_kernels()
        self._construct_transformations()

    def _construct_transformations(self):

        # TRANS: takes {name: mock_val} and returns necessary transformation code
        # for each value (in dict, I guess)
        c_trs = transformations_for_base(self._tr_tree, self._get_base_values())
        leaf_names = set(self._tr_tree.leaf_names())

        for operation in self._operations:
            if not isinstance(operation, KernelCall):
                continue

            transformation_code = self._tr_tree.transformations_for(operation.argnames)
            operation.set_transformations(transformation_code)

    def _render(self, template, **kwds):
        # Gets Mako template as a parameter (Template object)

        class PrefixHandler:

            def __init__(self, prefix=''):
                self.prefix = prefix

            def __getattr__(self, name):
                return self.prefix + name

        dtypes_dict = AttrDict(self._get_base_values())
        ctypes_dict = AttrDict({name:ctype(dtype) for name, dtype in dtypes.items()})

        # TODO: check for errors in load/stores/param usage?
        # TODO: add some more "built-in" variables (helpers, cluda.dtypes)?
        render_kwds = dict(
            basis=self._basis,
            load=PrefixHandler('_LOAD_'), # TODO: remove hardcoding (transformations.py uses it too)
            store=PrefixHandler('_STORE_'), # TODO: remove hardcoding (transformations.py uses it too)
            param=PrefixHandler(),
            ctype=ctypes_dict,
            dtype=dtypes_dict,
            signature='SIGNATURE')

        # check that user keywords do not overlap with our keywords
        assert set(render_kwds).isdisjoint(set(kwds))

        render_kwds.update(kwds)
        return render(template, **render_kwds)

    def connect(self, tr, endpoint, new_endpoints, new_scalar_endpoints=None):
        if new_scalar_endpoints is None: new_scalar_endpoints = []

        # TRANS: checks that there are no nodes named like this in a tree
        assert not self._tr_tree.has_nodes(*(new_endpoints + new_scalar_endpoints))
        # TRANS: checks that there is no endpoint named like this
        if self._tr_tree.has_endpoint(endpoint):
            # TRANS: connect given transformation to endpoint
            self._tr_tree.connect(tr, endpoint, new_endpoints, new_scalar_endpoints)
        else:
            raise Exception("Endpoint " + endpoint + " was not found")

        self._construct_transformations()

    def prepare(self, **kwds):
        if self._basis_needs_update(**kwds):
            self._change_basis(**kwds)
        return self

    def prepare_for(self, *args):
        new_basis = self._basis_for(*args)
        return self.prepare(**new_basis)

    @property
    def signature(self):
        return self._tr_tree.outward_signature()

    def __call__(self, *args):
        if self._debug:
            new_basis = self._basis_for(*args):
            if self._basis_needs_update(**new_basis):
                raise Exception("Given arguments require different basis")

        # TODO: profile this and see if it's a bottleneck
        signature = self._tr_tree.outward_signature()
        arg_dict = {}
        for pair, arg in zip(signature, args):
            name, _ = pair
            # TODO: check types here if _debug is on
            arg_dict[name] = arg

        # TODO: add internally allocated arrays to arg_dict
        # TODO: how to handle external calls, like Transpose in Reduce?
        #   (solution: we request the same execution list from Transpose,
        #   set argument names - should be a method for that - and incorporate it into our own list)
        # TODO: cool feature: process the list and remove unnecessary allocations,
        # replacing them by creating views

        for operation in self._operations:
            op_args = [arg_dict[name] for name in operation.argnames]
            operation(*op_args)


class KernelCall:

    def __init__(self, name, argnames, base_src, block=(1,1,1), grid=(1,1)):
        self.name = name
        self.argnames = argnames
        self.block = block
        self.grid = grid
        self.src = base_src

    def __call__(self, *args):
        self.kernel(*args, block=self.block, grid=self.grid, shared=self.shared)




