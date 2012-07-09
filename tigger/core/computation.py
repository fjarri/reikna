import numpy
import os, os.path

product = lambda x: reduce(lambda x1, x2: x1 * x2, x, 1)


class AttrDict(dict):

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __repr__(self):
        return "AttrDict(" + dict.__repr__(self) + ")"

    def needsUpdate(self, other):
        for key in other:
            assert key in self, "Unknown key: " + key
            if self[key] != other[key]:
                return True

        return False


def loadTemplateFor(filename):
    name, ext = os.path.splitext(filename)
    template = name + ".cu.mako"
    return open(template).read()


def strip_array(arr):
    fields = ['shape', 'size', 'dtype']
    return AttrDict().update({key:getattr(arr, key) for key in fields})


class Computation:

    def __init__(self, env, debug=False):
        self._env = env
        self._debug = debug

        # DERIVED: returns {} with basis parameters
        self._basis = AttrDict(**self._get_default_basis())

        # DERIVED: returns [name], [], []
        stores, loads, scalars = self._get_base_names()

        # TRANS: takes [], [], [] with endpoint names
        self._tr_tree = TransformationTree(stores, loads, scalars)

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

        if self._basis.needsUpdate(kwds):
            self._basis.update(kwds)

        # DERIVED: returns {name: mock_val}
        values_dict = self._get_base_values()

        # TRANS: takes {name: mock_val} and propagates it from roots to leaves
        self._tr_tree.propagate_outward(values_dict)
        # need to somehow pass kernel creator the following things:
        # - outward signature (to be inserted to the kernel)
        # - generated transformation functions and macros
        self._derived = self._construct_derived()

        return self

    def prepare_for(self, *args):

        # TRANS: returns [name], [], [] with endpoint names
        out_stores, out_loads, out_scalars = self._tr_tree.outward_names()
        names = out_stores + out_loads + out_scalars

        assert len(args) == len(names)

        values = {}
        for name, arg for zip(names, args)
            # TODO: check that arg has correct type (array/scalar)
            if hasattr(arg, 'shape'):
                types_dict[name] = MockValue.array(arg)
            else:
                types_dict[name] = MockValue.scalar(arg)

        # TRANS: takes {name: mock_val} and propagates it from leaves to roots
        self._tr_tree.propagate_inward(types_dict)

        new_args = self._tr_tree.inward_signature()
        b = self._construct_basis(*new_args)

        return self.prepare(**b)

    def _construct_basis(self, *args):
        return AttrDict()

    def _construct_derived(self):
        return AttrDict()

    def _set_kernels(self, kernel_calls):
        self._kernels = kernel_calls

    @property
    def signature(self):
        return self._tr_tree.outward_signature()

    def __call__(self, *args):
        if self._debug:

            # do the same as in prepare_for, but checking if the preparation was necessary

            bs = self._construct_basis(*args)
            if self._basis.needsUpdate(bs):
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


class Transformation:

    def __init__(self, load=1, store=1, parameters=0,
            derive_store=lambda *x: x[0],
            derive_load=lambda *x: x[0],
            code="store1(load1)"):
        self.load = load
        self.store = store
        self.parameters = parameters
        self.derive_store = derive_store
        self.derive_load = derive_load

        # TODO: run code through Mako and check that number of load/store/parameters match
        self.code = code


def min_blocks(length, block):
    return (length - 1) / block + 1

def log2(n):
    pos = 0
    for pow in [16, 8, 4, 2, 1]:
        if n >= 2 ** pow:
            n /= (2 ** pow)
            pos += pow
    return pos


