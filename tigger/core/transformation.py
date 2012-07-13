import numpy
from tigger.cluda.dtypes import cast
from tigger.core.helpers import AttrDict


LOAD_PREFIX = '_LOAD_'
STORE_PREFIX = '_STORE_'
SIGNATURE = 'SIGNATURE'

class ArrayValue:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.is_array = True

    def __repr__(self):
        return "ArrayValue(" + repr(self.shape) + "," + repr(self.dtype) + ")"


class ScalarValue:
    def __init__(self, value, dtype):
        self.value = cast(dtype)(value) if value is not None else value
        self.dtype = dtype
        self.is_array = False

    def __repr__(self):
        return "ScalarValue(" + repr(self.value) + "," + repr(self.dtype) + ")"


def wrap_value(value):
    if hasattr(value, 'shape'):
        return ArrayValue(value.shape, value.dtype)
    else:
        dtype = numpy.min_scalar_type(value)
        return ScalarValue(value, dtype)


class Transformation:

    def __init__(self, load=1, store=1, parameters=0,
            derive_s_from_lp=None,
            derive_lp_from_s=None,
            derive_l_from_sp=None,
            derive_sp_from_l=None,
            code="${store.s1}(${load.l1});"):
        self.load = load
        self.store = store
        self.parameters = parameters

        def get_derivation_func(return_tuple, l1, l2=0):
            def func(*x):
                dtype = numpy.result_type(*x)
                if return_tuple:
                    return [dtype] * l1, [dtype] * l2
                else:
                    return [dtype] * l1
            return func

        if derive_s_from_lp is None: derive_s_from_lp = get_derivation_func(False, store)
        if derive_lp_from_s is None: derive_lp_from_s = get_derivation_func(True, load, parameters)
        if derive_l_from_sp is None: derive_l_from_sp = get_derivation_func(False, load)
        if derive_sp_from_l is None: derive_sp_from_l = get_derivation_func(True, store, parameters)

        self.derive_s_from_lp = derive_s_from_lp
        self.derive_lp_from_s = derive_lp_from_s
        self.derive_l_from_sp = derive_l_from_sp
        self.derive_sp_from_l = derive_sp_from_l

        # TODO: run code through Mako and check that number of load/store/parameters match
        self.code = code


NODE_LOAD = 0
NODE_STORE = 1
NODE_SCALAR = 2


class TransformationTree:

    def __init__(self, stores, loads, scalars):
        self.nodes = {}
        # TODO: check for repeating names?
        for name in stores:
            self.nodes[name] = AttrDict(name=name, type=NODE_STORE,
                value=ArrayValue(None, None),
                parent=None, children=None, tr_to_parent=None, tr_to_children=None)
        for name in loads:
            self.nodes[name] = AttrDict(name=name, type=NODE_LOAD,
                value=ArrayValue(None, None),
                parent=None, children=None, tr_to_parent=None, tr_to_children=None)
        for name in scalars:
            self.nodes[name] = AttrDict(name=name, type=NODE_SCALAR,
                value=ScalarValue(None, None),
                parent=None, children=None, tr_to_parent=None, tr_to_children=None)

        self.base_names = stores + loads + scalars

    def leaf_signature(self):
        visited = set()
        arrays = []
        scalars = [name for name in self.base_names if not self.nodes[name].value.is_array]

        def visit(names):
            for name in names:
                if name in visited: continue
                visited.add(name)
                node = self.nodes[name]
                if node.children is None:
                    arrays.append(name)
                else:
                    array_children = [name for name in node.children if self.nodes[name].value.is_array]
                    visit(array_children)
                    scalar_children = [name for name in node.children if not self.nodes[name].value.is_array]
                    scalars.extend(scalar_children)

        array_names = [name for name in self.base_names if self.nodes[name].value.is_array]
        visit(array_names)
        return [(name, self.nodes[name].value) for name in arrays + scalars]

    def _clear_values(self):
        for name in self.nodes:
            old_value = self.nodes[name].value
            # FIXME: Creating new values in case the old ones are copies
            if old_value.is_array:
                self.nodes[name].value = ArrayValue(old_value.shape, None)
            else:
                self.nodes[name].value = ScalarValue(old_value.value, None)

    def propagate_to_base(self, values_dict):
        # takes {name: mock_val} and propagates it from leaves to roots,
        # updating nodes

        self._clear_values()

        def deduce(name):
            node = self.nodes[name]
            if node.children is None:
                node.value = values_dict[name]
                return

            for child in node.children:
                deduce(child)

            child_dtypes = [self.nodes[child].value.dtype for child in node.children]
            tr = node.tr_to_children
            derive_types = tr.derive_l_from_sp if node.type == NODE_STORE else tr.derive_s_from_lp
            node.value.dtype = derive_types(*child_dtypes)[0]

        for name in self.base_names:
            deduce(name)

    def propagate_to_leaves(self, values_dict):
        # takes {name: mock_val} and propagates it from roots to leaves,
        # updating nodes

        self._clear_values()

        def propagate(name):
            node = self.nodes[name]
            if node.children is None:
                return

            tr = node.tr_to_children
            derive_types = tr.derive_sp_from_l if node.type == NODE_STORE else tr.derive_lp_from_s
            arr_dtypes, scalar_dtypes = derive_types(node.value.dtype)
            for child, dtype in zip(node.children, arr_dtypes + scalar_dtypes):
                child_value = self.nodes[child].value
                if child_value.dtype is None:
                    child_value.dtype = dtype
                elif child_value.dtype != dtype:
                    raise Exception("Data type conflict in node " + child +
                        " while propagating types to leaves")
                propagate(child)

        for name in self.base_names:
            self.nodes[name].value = values_dict[name]
            propagate(name)


    def transformations_for(self, names):
        # takes [name] for bases and returns necessary transformation code
        # if some of the names are not in base, they are treated as leaves
        # returns string with all the transformation code
        pass

    def has_nodes(self, names):
        for name in names:
            if name in self.nodes:
                return True
        return False

    def has_array_leaf(self, name):
        names = set(n for n, v in self.leaf_signature() if v.is_array)
        return name in names

    def connect(self, tr, endpoint, new_endpoints, new_scalar_endpoints=None):
        parent = self.nodes[endpoint]
        if new_scalar_endpoints is None:
            new_scalar_endpoints = []
        parent.children = new_endpoints + new_scalar_endpoints
        parent.tr_to_children = tr

        # TODO: check that the transformation only has one load/store
        # in the place where connection occurs

        for ep in new_endpoints:
            self.nodes[ep] = AttrDict(name=ep, type=parent.type,
                value=ArrayValue(None, None),
                parent=parent.name, children=None, tr_to_parent=tr, tr_to_children=None)
        for ep in new_scalar_endpoints:
            self.nodes[ep] = AttrDict(name=ep, type=NODE_SCALAR,
                value=ScalarValue(None, None),
                parent=parent.name, children=None, tr_to_parent=tr, tr_to_children=None)


if __name__ == '__main__':

    a = Transformation(load=1, store=1,
        code="${store.s1}(${load.l1});")

    b = Transformation(load=2, store=1, parameters=1,
        derive_s_from_lp=lambda l1, l2, p1: [l1],
        derive_lp_from_s=lambda s1: ([s1, s1], [numpy.int32]),
        derive_l_from_sp=lambda s1, p1: [s1, s1],
        derive_sp_from_l=lambda l1, l2: ([l1], [numpy.int32]),
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

    tree = TransformationTree(['C'], ['A', 'B'], ['coeff'])
    print tree.leaf_signature()

    tree.connect(a, 'A', ['A_prime']);
    tree.connect(b, 'B', ['A_prime', 'B_prime'], ['B_param'])
    tree.connect(a, 'B_prime', ['B_new_prime'])
    tree.connect(c, 'C', ['C_half1', 'C_half2'])
    tree.connect(a, 'C_half1', ['C_new_half1'])
    print tree.leaf_signature()
    print tree.has_array_leaf('A_prime') # True
    print tree.has_array_leaf('coeff') # False
    print tree.has_nodes(['C_half1']) # True

    tree.propagate_to_leaves(dict(
        A=ArrayValue(None, numpy.float32),
        B=ArrayValue(None, numpy.float32),
        C=ArrayValue(None, numpy.float32),
        coeff=ScalarValue(None, numpy.float32)
    ))
    print tree.leaf_signature()

    tree.propagate_to_base(dict(
        A_prime=ArrayValue(None, numpy.float64),
        B_new_prime=ArrayValue(None, numpy.float32),
        C_half2=ArrayValue(None, numpy.float64),
        C_new_half1=ArrayValue(None, numpy.float64),
        coeff=ScalarValue(None, numpy.float32),
        B_param=ScalarValue(None, numpy.float64)
    ))
    print tree.leaf_signature()

    print tree.transformations_for(['C', 'A', 'B', 'coeff'])
