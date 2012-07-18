import numpy
from mako.template import Template

import tigger.cluda.dtypes as dtypes
from tigger.cluda.kernel import render_without_funcs, FuncCollector
from tigger.core.helpers import AttrDict, product


INDEX_NAME = "idx"
VALUE_NAME = "val"


def load_macro_name(name):
    return "_LOAD_" + name

def load_function_name(name):
    return "_load_" + name

def leaf_load_macro(name):
    return "#define {macro_name} ({name}[{idx}])".format(
        macro_name=load_macro_name(name), name=name,
        idx=INDEX_NAME, val=VALUE_NAME)

def node_load_macro(name, argnames):
    return "#define {macro_name} {fname}({arglist}, {idx})".format(
        macro_name=load_macro_name(name), fname=load_function_name(name),
        arglist = ", ".join(argnames), idx=INDEX_NAME, val=VALUE_NAME)

def base_leaf_load_macro(name):
    return "#define {macro_name}({idx}) ({name}[{idx}])".format(
        macro_name=load_macro_name(name), name=name,
        idx=INDEX_NAME, val=VALUE_NAME)

def base_node_load_macro(name, argnames):
    return "#define {macro_name}({idx}) {fname}({arglist}, {idx})".format(
        macro_name=load_macro_name(name), fname=load_function_name(name),
        arglist = ", ".join(argnames), idx=INDEX_NAME, val=VALUE_NAME)

def load_macro_call(name):
    return load_macro_name(name)



def store_macro_name(name):
    return "_STORE_" + name

def store_function_name(name):
    return "_store_" + name

def leaf_store_macro(name):
    return "#define {macro_name}({val}) {name}[{idx}] = ({val})".format(
        macro_name=store_macro_name(name), name=name,
        idx=INDEX_NAME, val=VALUE_NAME)

def node_store_macro(name, argnames):
    return "#define {macro_name}({val}) {fname}({arglist}, {idx}, {val})".format(
        macro_name=store_macro_name(name), fname=store_function_name(name),
        arglist = ", ".join(argnames), idx=INDEX_NAME, val=VALUE_NAME)

def base_leaf_store_macro(name):
    return "#define {macro_name}({idx}, {val}) {name}[{idx}] = ({val})".format(
        macro_name=store_macro_name(name), name=name,
        idx=INDEX_NAME, val=VALUE_NAME)

def base_node_store_macro(name, argnames):
    return "#define {macro_name}({idx}, {val}) {fname}({arglist}, {idx}, {val})".format(
        macro_name=store_macro_name(name), fname=store_function_name(name),
        arglist = ", ".join(argnames), idx=INDEX_NAME, val=VALUE_NAME)

def store_macro_call(name):
    return store_macro_name(name)


def signature_macro_name():
    return "SIGNATURE"



class ArrayValue:
    def __init__(self, shape, dtype):
        self.shape = shape
        if shape is not None:
            self.size = product(shape)
        self.dtype = dtype
        self.is_array = True

    def __repr__(self):
        return "ArrayValue(" + repr(self.shape) + "," + repr(self.dtype) + ")"


class ScalarValue:
    def __init__(self, value, dtype):
        self.value = dtypes.cast(dtype)(value) if value is not None else value
        self.dtype = dtype
        self.is_array = False

    def __repr__(self):
        return "ScalarValue(" + repr(self.value) + "," + repr(self.dtype) + ")"


def wrap_value(value):
    if hasattr(value, 'shape') and len(value.shape) > 0:
        return ArrayValue(value.shape, value.dtype)
    else:
        dtype = dtypes.min_scalar_type(value)
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
                dtype = dtypes.result_type(*x)
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

        self.code = Template(code)


NODE_LOAD = 0
NODE_STORE = 1
NODE_SCALAR = 2


class TransformationTree:

    def __init__(self, stores, loads, scalars):
        self.nodes = {}
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

    def leaf_signature(self, base_name=None):
        visited = set()
        arrays = []

        if base_name is None:
            scalars = [name for name in self.base_names if not self.nodes[name].value.is_array]
        else:
            scalars = []

        def visit(names):
            for name in names:
                if name in visited: continue
                visited.add(name)
                node = self.nodes[name]
                if node.children is None:
                    arrays.append(name)
                else:
                    array_children = [name for name in node.children
                        if self.nodes[name].value.is_array]
                    visit(array_children)
                    scalar_children = [name for name in node.children
                        if not self.nodes[name].value.is_array]
                    scalars.extend(scalar_children)

        if base_name is None:
            array_names = [name for name in self.base_names if self.nodes[name].value.is_array]
            visit(array_names)
        else:
            visit([base_name])

        return [(name, self.nodes[name].value) for name in arrays + scalars]

    def base_values(self):
        return [self.nodes[name].value for name in self.base_names]

    def all_children(self, name):
        return [name for name, _ in self.leaf_signature(name)]

    def _clear_values(self):
        for name in self.nodes:
            old_value = self.nodes[name].value
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

            # derive type
            child_dtypes = [self.nodes[child].value.dtype for child in node.children]
            tr = node.tr_to_children
            derive_types = tr.derive_l_from_sp if node.type == NODE_STORE else tr.derive_s_from_lp
            node.value.dtype = derive_types(*child_dtypes)[0]

            # derive shape
            child_shapes = [self.nodes[child].value.shape for child in node.children
                if hasattr(self.nodes[child].value, 'shape')]
            assert len(set(child_shapes)) == 1
            node.value.shape = child_shapes[0]

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
        visited = set()
        code_list = []
        func_c = FuncCollector(prefix="tr")

        def build_arglist(argnames):
            res = []
            for argname in argnames:
                value = self.nodes[argname].value
                dtype = self.nodes[argname].value.dtype
                ctype = dtypes.ctype(dtype)
                res.append(("GLOBAL_MEM " if value.is_array else " ") +
                    ctype + (" *" if value.is_array else " ") + argname)

            return ", ".join(res)

        def signature_macro(argnames):
            res = []
            for argname in argnames:
                value = self.nodes[argname].value
                dtype = self.nodes[argname].value.dtype
                ctype = dtypes.ctype(dtype)
                res.append(("GLOBAL_MEM " if value.is_array else " ") +
                    ctype + (" *" if value.is_array else " ") + argname)

            return "#define {macro_name} {arglist}".format(
                macro_name=signature_macro_name(),
                arglist=", ".join(res))

        def process(name):
            if name in visited: return

            visited.add(name)
            node = self.nodes[name]

            if node.type == NODE_LOAD:
                if name in self.base_names:
                    leaf_macro = base_leaf_load_macro
                    node_macro = base_node_load_macro
                else:
                    leaf_macro = leaf_load_macro
                    node_macro = node_load_macro
            elif node.type == NODE_STORE:
                if name in self.base_names:
                    leaf_macro = base_leaf_store_macro
                    node_macro = base_node_store_macro
                else:
                    leaf_macro = leaf_store_macro
                    node_macro = node_store_macro
            else:
                return

            if node.children is None:
                code_list.append("// leaf node " + node.name + "\n" + leaf_macro(node.name))
                return

            for child in node.children:
                process(child)

            all_children = self.all_children(node.name)
            tr = node.tr_to_children

            if node.type == NODE_LOAD:
                definition = "INLINE WITHIN_KERNEL {outtype} {fname}({arglist}, int idx)".format(
                    outtype=dtypes.ctype(node.value.dtype),
                    fname=load_function_name(node.name),
                    arglist=build_arglist(all_children))
                load_names = node.children[:tr.load]
                param_names = node.children[tr.load:]

                load = AttrDict()
                param = AttrDict()
                dtype = AttrDict()
                for i, name in enumerate(load_names):
                    label = 'l' + str(i+1)
                    load[label] = load_macro_call(name)
                    dtype[label] = self.nodes[name].value.dtype
                for i, name in enumerate(param_names):
                    label = 'p' + str(i+1)
                    param[label] = name
                    dtype[label] = self.nodes[name].value.dtype

                store = AttrDict(s1='return')
                dtype.s1 = node.value.dtype
            else:
                definition = "INLINE WITHIN_KERNEL void {fname}({arglist}, int idx, {intype} val)".format(
                    intype=dtypes.ctype(node.value.dtype),
                    fname=store_function_name(node.name),
                    arglist=build_arglist(all_children))

                store_names = node.children[:tr.store]
                param_names = node.children[tr.store:]

                store = AttrDict()
                dtype = AttrDict()
                param = AttrDict()
                for i, name in enumerate(store_names):
                    label = 's' + str(i+1)
                    store[label] = store_macro_call(name)
                    dtype[label] = self.nodes[name].value.dtype
                for i, name in enumerate(param_names):
                    label = 'p' + str(i+1)
                    param[label] = name
                    dtype[label] = self.nodes[name].value.dtype

                load = AttrDict(l1='val')
                dtype.l1 = node.value.dtype

            ctype = AttrDict({key:dtypes.ctype(dt) for key, dt in dtype.items()})

            code_src = render_without_funcs(tr.code, func_c,
                load=load, store=store, dtype=dtype, ctype=ctype, param=param)

            code_list.append("// node " + node.name + "\n" +
                definition + "\n{\n" + code_src + "\n}\n" +
                node_macro(node.name, all_children))

        for name in names:
            if name in self.base_names:
                process(name)
            else:
                code_list.append(load_macro(name))
                code_list.append(store_macro(name))

        leaf_names = [name for name, _ in self.leaf_signature()]
        return func_c.render() + "\n\n" + "\n\n".join(code_list) + "\n\n" + signature_macro(leaf_names)

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

        for ep in new_endpoints:
            self.nodes[ep] = AttrDict(name=ep, type=parent.type,
                value=ArrayValue(None, None),
                parent=parent.name, children=None, tr_to_parent=tr, tr_to_children=None)
        for ep in new_scalar_endpoints:
            self.nodes[ep] = AttrDict(name=ep, type=NODE_SCALAR,
                value=ScalarValue(None, None),
                parent=parent.name, children=None, tr_to_parent=tr, tr_to_children=None)
