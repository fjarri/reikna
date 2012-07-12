import numpy
from tigger.cluda.dtypes import cast


LOAD_PREFIX = '_LOAD_'
STORE_PREFIX = '_STORE_'
SIGNATURE = 'SIGNATURE'

class ArrayValue:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.is_array = True


class ScalarValue:
    def __init__(self, value, dtype):
        self.value = cast(dtype)(value) if values is not None else value
        self.dtype = dtype
        self.is_array = False


def wrap_value(value):
    if hasattr(value, 'shape'):
        return ArrayValue(value.shape, value.dtype)
    else:
        dtype = numpy.min_scalar_type(value)
        return ScalarValue(value, dtype)


class Transformation:

    def __init__(self, load=1, store=1, parameters=0,
            derive_store=lambda *x: x[0],
            derive_load=lambda *x: x[0],
            code="${store.s1}(${load.l1});"):
        self.load = load
        self.store = store
        self.parameters = parameters
        self.derive_store = derive_store
        self.derive_load = derive_load

        # TODO: run code through Mako and check that number of load/store/parameters match
        self.code = code


class TransformationTree:

    def __init__(self, loads, stores, scalars):
        pass

    def leaf_signature(self):
        # returns [(name, mock_value)]
        pass

    def propagate_to_base(self, values_dict):
        # takes {name: mock_val} and propagates it from leaves to roots,
        # updating nodes

    def propagate_to_leaves(self, values_dict):
        # takes {name: mock_val} and propagates it from roots to leaves,
        # updating nodes

    def transformations_for(self, names):
        # takes [name] for bases and returns necessary transformation code
        # if some of the names are not in base, they are treated as leaves
        # returns string with all the transformation code

    def has_nodes(self, *names):
        # checks that there are no nodes named like this in the whole tree
        pass

    def has_leaf(self, name):
        # checks that there is no leaf named like this
        pass

    def connect(self, tr, endpoint, new_endpoints, new_scalar_endpoints):
        # connect given transformation to endpoint
        pass


if __name__ == '__main__':

    a = Transformation(load=1, store=1,
        code="${store.s1}(${load.l1});")

    b = Transformation(load=2, store=1, parameters=1,
        derive_store=lambda t1, _: ([t1], [numpy.int32]),
        derive_load=lambda t1: ([t1, t1], [numpy.int32]),
        code="""
            ${ctype.s1} t = ${mul(dtype.s1, dtype.l1)}(${load.p1}, ${load.l1});
            ${store.s1}(t + ${load.l2});
        """)

    c = Transformation(load=1, store=2,
        code="""
            ${ctype.s1} t = ${mul(dtype.l1, float32)}(${load.l1}, 0.5);
            ${store.s1}(t);
            ${store.s2}(t);
        """)

    tree = TransformationTree.stores(['C'], ['A', 'B'], ['coeff'])

    tree.connect(a, 'A', ['A_prime']);
    tree.connect(b, 'B', ['A_prime', 'B_prime'], ['B_param'])
    tree.connect(a, 'B_prime', ['B_new_prime'])
    tree.connect(c, 'C', ['C_half1', 'C_half2'])
    tree.connect(a, 'C_half1', ['C_new_half1'])

    print tree.transformations_for(['C', 'A', 'B', 'coeff'])
