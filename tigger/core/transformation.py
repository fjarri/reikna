
def strip_array(arr):
    fields = ['shape', 'size', 'dtype']
    return AttrDict().update({key:getattr(arr, key) for key in fields})


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

    def __init__(self, *roots):
        pass

    @classmethod
    def loads(cls, *roots):
        pass

    @classmethod
    def stores(cls, *roots):
        pass

    def has_nodes(self, *names):
        pass

    def has_endpoint(self, name):
        pass

    def connect(self, tr, endpoint, new_endpoints):
        pass

    def propagate_inward(self, types_dict):
        pass

    def propagate_outward(self, types_dict):
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

    out_tree = TransformationTree.stores('C')
    in_tree = TransformationTree.loads('A', 'B')

    in_tree.connect(a, 'A', ['A_prime']);
    in_tree.connect(b, 'B', ['A_prime', 'B_prime'], ['B_param'])
    in_tree.connect(a, 'B_prime', ['B_new_prime'])
    out_tree.connect(c, 'C', ['C_half1', 'C_half2'])
    out_tree.connect(a, 'C_half1', ['C_new_half1'])



