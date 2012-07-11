import numpy
import os, os.path


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

        # TRANS: takes {name: mock_val} and returns necessary transformation code
        # for each value (in dict, I guess)
        c_trs = transformations_for_base(self._tr_tree, self._get_base_values())
        # need to somehow pass kernel creator the following things:
        # - outward signature (to be inserted to the kernel)
        # - generated transformation functions and macros
        self._derived = self._construct_derived()

    def connect(self, tr, endpoint, new_endpoints, new_scalar_endpoints=None):
        if scalar_endpoints is None: scalar_endpoints = []

        # TRANS: checks that there are no nodes named like this in a tree
        assert not self._tr_tree.has_nodes(*(new_endpoints + new_scalar_endpoints))
        # TRANS: checks that there is no endpoint named like this
        if self._tr_tree.has_endpoint(endpoint):
            # TRANS: connect given transformation to endpoint
            self._tr_tree.connect(tr, endpoint, new_endpoints, new_scalar_endpoints)
        else:
            raise Exception("Endpoint " + endpoint + " was not found")

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

        # Variant1: call _call() of derived class and let it handle kernel calls
        #
        # Variant2:
        # get current signature
        # assign arguments to the dictionary dict(A_prime=args[0], ...)
        # process list created by _construct_derived(), which can contain:
        # - allocate <argname>, ...
        # - call <kernelname>, <arglist>,
        # - something to handle external calls, like Transpose in Reduce?
        #   (solution: we request the same execution list from Transpose,
        #   set argument names - should be a method for that - and incorporate it into our own list)
        # For each command, get corresponding args from the dictionary and execute it
        #
        # Is there anything that cannot be described by variant 2?
        # If no, then this can be later decoupled from actual execution
        #
        # Cool feature: process the list and remove unnecessary allocations,
        # replacing them by creating views

        for kernel_call in self._kernels:
            kernel_call(*args)

        # reduction: (dest, src)
        # - allocate tmp
        # - reduce1024(tmp, src)
        # - reduce32(dest, tmp)


class KernelCall:

    def __init__(self, block=None, grid=None, kernel=None, shared=0):
        self.block = block
        self.grid = grid
        self.kernel = kernel
        self.shared = shared
        # TODO: prepare kernel here?

    def __call__(self, *args):
        self.kernel(*args, block=self.block, grid=self.grid, shared=self.shared)




