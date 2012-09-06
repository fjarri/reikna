import re

import numpy
from mako.template import Template

import tigger.cluda.dtypes as dtypes
from tigger.cluda.kernel import render_without_funcs, FuncCollector
from tigger.helpers import AttrDict, product


INDEX_NAME = "idx"
VALUE_NAME = "val"


class TypePropagationError(Exception):
    pass


def load_macro_name(name):
    return "_LOAD_" + name

def load_function_name(name):
    return "_load_" + name

def leaf_load_macro(name):
    return "#define {macro_name}({idx}) ({name}[{idx}])".format(
        macro_name=load_macro_name(name), name=leaf_name(name),
        idx=INDEX_NAME, val=VALUE_NAME)

def node_load_macro(name, argnames):
    return "#define {macro_name}({idx}) {fname}({arglist}, {idx})".format(
        macro_name=load_macro_name(name), fname=load_function_name(name),
        arglist = ", ".join([leaf_name(name) for name in argnames]),
        idx=INDEX_NAME, val=VALUE_NAME)

def base_leaf_load_macro(name):
    return "#define {macro_name}({idx}) ({name}[{idx}])".format(
        macro_name=load_macro_name(name), name=leaf_name(name),
        idx=INDEX_NAME, val=VALUE_NAME)

def base_node_load_macro(name, argnames):
    return "#define {macro_name}({idx}) {fname}({arglist}, {idx})".format(
        macro_name=load_macro_name(name), fname=load_function_name(name),
        arglist = ", ".join([leaf_name(name) for name in argnames]),
        idx=INDEX_NAME, val=VALUE_NAME)

def load_macro_call(name):
    return load_macro_name(name)

def load_macro_call_tr(name):
    return "{macro_name}({idx})".format(
        macro_name=load_macro_name(name), idx=INDEX_NAME)



def store_macro_name(name):
    return "_STORE_" + name

def store_function_name(name):
    return "_store_" + name

def leaf_store_macro(name):
    return "#define {macro_name}({val}) {name}[{idx}] = ({val})".format(
        macro_name=store_macro_name(name), name=leaf_name(name),
        idx=INDEX_NAME, val=VALUE_NAME)

def node_store_macro(name, argnames):
    return "#define {macro_name}({val}) {fname}({arglist}, {idx}, {val})".format(
        macro_name=store_macro_name(name), fname=store_function_name(name),
        arglist = ", ".join([leaf_name(name) for name in argnames]),
        idx=INDEX_NAME, val=VALUE_NAME)

def base_leaf_store_macro(name):
    return "#define {macro_name}({idx}, {val}) {name}[{idx}] = ({val})".format(
        macro_name=store_macro_name(name), name=leaf_name(name),
        idx=INDEX_NAME, val=VALUE_NAME)

def base_node_store_macro(name, argnames):
    return "#define {macro_name}({idx}, {val}) {fname}({arglist}, {idx}, {val})".format(
        macro_name=store_macro_name(name), fname=store_function_name(name),
        arglist = ", ".join([leaf_name(name) for name in argnames]),
        idx=INDEX_NAME, val=VALUE_NAME)

def store_macro_call(name):
    return store_macro_name(name)


def signature_macro_name():
    return "SIGNATURE"

def kernel_definition(kernel_name):
    return "KERNEL void {kernel_name}(SIGNATURE)".format(kernel_name=kernel_name)

def leaf_name(name):
    return "_leaf_" + name


def valid_argument_name(name):
    return (re.match(r"^[a-zA-Z_]\w*$", name) is not None)


class ArrayValue(object):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtypes.normalize_type(dtype) if dtype is not None else None
        self.is_array = True

    def fill_with(self, other):
        self.shape = other.shape
        self.dtype = other.dtype

    def clear(self):
        self.shape = None
        self.dtype = None

    def get_shape(self):
        return self._shape

    def set_shape(self, shape):
        self._shape = shape
        if shape is None:
            self._size = None
        else:
            self._size = product(shape)

    shape = property(get_shape, set_shape)

    @property
    def size(self):
        return self._size

    def __str__(self):
        props = ["array"]
        if self.dtype is not None:
            props.append(str(self.dtype))
        if self.shape is not None:
            props.append(str(self.shape))
        return ", ".join(props)

    def __repr__(self):
        return "ArrayValue(" + repr(self.shape) + "," + repr(self.dtype) + ")"


class ScalarValue:
    def __init__(self, value, dtype):
        self.value = dtypes.cast(dtype)(value) if value is not None else value
        self.dtype = dtypes.normalize_type(dtype) if dtype is not None else None
        self.is_array = False

    def fill_with(self, other):
        self.value = other.value
        self.dtype = other.dtype

    def clear(self):
        self.value = None
        self.dtype = None

    def __str__(self):
        props = ["scalar"]
        if self.dtype is not None:
            props.append(str(self.dtype))
        return ", ".join(props)

    def __repr__(self):
        return "ScalarValue(" + repr(self.value) + "," + repr(self.dtype) + ")"


def wrap_value(value):
    if hasattr(value, 'shape') and len(value.shape) > 0:
        return ArrayValue(value.shape, value.dtype)
    else:
        dtype = dtypes.min_scalar_type(value)
        return ScalarValue(value, dtype)


class Transformation:
    """
    Defines an elementwise transformation.

    :param inputs: number of input array values.
    :param outputs: number of output array values.
    :param parameters: number of scalar parameters for the transformation.
    :param derive_o_from_is: a function taking ``inputs`` + ``scalars`` dtype parameters
        and returning list with ``outputs`` dtypes.
        Used to derive types in the transformation tree after call to
        :py:meth:`Computation.prepare_for` when the transformation is connected
        to the input argument.
    :param derive_is_from_o: a function taking ``outputs`` dtype parameters
        and returning tuple of two lists with ``inputs`` and ``scalars`` dtypes.
        Used to derive types in the transformation tree after call to
        :py:meth:`Computation.prepare` when the transformation is connected to the input argument.
    :param derive_i_from_os: a function taking ``outputs`` + ``scalars`` dtype parameters
        and returning list with ``inputs`` dtypes.
        Used to derive types in the transformation tree after call to
        :py:meth:`Computation.prepare_for` when the transformation is connected
        to the output argument.
    :param derive_os_from_i: a function taking ``inputs`` dtype parameters
        and returning tuple of two lists with ``outputs`` and ``scalars`` dtypes.
        Used to derive types in the transformation tree after call to
        :py:meth:`Computation.prepare` when the transformation is connected to the output argument.
    :param code: template source with the transformation code.
        See :ref:`guide-writing-a-transformation` section for details.
    """

    def __init__(self, inputs=1, outputs=1, scalars=0,
            derive_o_from_is=None,
            derive_is_from_o=None,
            derive_i_from_os=None,
            derive_os_from_i=None,
            code="${o1.store}(${i1.load});"):
        self.inputs = inputs
        self.outputs = outputs
        self.scalars = scalars

        def get_derivation_func(return_tuple, n1, n2=0):
            def func(*x):
                dtype = dtypes.result_type(*x)
                if return_tuple:
                    return [dtype] * n1, [dtype] * n2
                else:
                    return [dtype] * n1
            return func

        if derive_o_from_is is None: derive_o_from_is = get_derivation_func(False, outputs)
        if derive_is_from_o is None: derive_is_from_o = get_derivation_func(True, inputs, scalars)
        if derive_i_from_os is None: derive_i_from_os = get_derivation_func(False, inputs)
        if derive_os_from_i is None: derive_os_from_i = get_derivation_func(True, outputs, scalars)

        self.derive_o_from_is = derive_o_from_is
        self.derive_is_from_o = derive_is_from_o
        self.derive_i_from_os = derive_i_from_os
        self.derive_os_from_i = derive_os_from_i

        self.code = Template(code)


NODE_INPUT = 0
NODE_OUTPUT = 1
NODE_SCALAR = 2


class TransformationArgument:

    def __init__(self, nodes, kind, number, name, node_type):
        self.label = kind + str(number + 1)
        self._name = name
        self.dtype = nodes[self._name].value.dtype
        self.ctype = dtypes.ctype(self.dtype)

        if node_type == NODE_INPUT:
            if kind == 'i':
                self.load = load_macro_call_tr(self._name)
            elif kind == 'o':
                self.store = "return"
        else:
            if kind == 'i':
                self.load = "val"
            elif kind == 'o':
                self.store = store_macro_call(self._name)

    def __str__(self):
        return leaf_name(self._name)


class TransformationTree:

    def __init__(self, outputs, inputs, scalars):
        self.nodes = {}
        self.temp_nodes = {}
        self.base_names = outputs + inputs + scalars

        # check names for correctness
        for name in self.base_names:
            if not valid_argument_name(name):
                raise ValueError("Incorrect argument name: " + name)

        # check for repeating names
        if len(set(self.base_names)) != len(self.base_names):
            raise ValueError("There are repeating argument names")

        for name in outputs:
            self.nodes[name] = AttrDict(name=name, type=NODE_OUTPUT,
                value=ArrayValue(None, None),
                children=None, tr_to_children=None)
        for name in inputs:
            self.nodes[name] = AttrDict(name=name, type=NODE_INPUT,
                value=ArrayValue(None, None),
                children=None, tr_to_children=None)
        for name in scalars:
            self.nodes[name] = AttrDict(name=name, type=NODE_SCALAR,
                value=ScalarValue(None, None),
                children=None, tr_to_children=None)

    def leaf_signature(self, base_names=None):

        if base_names is None:
            base_names = self.base_names

        arrays = []

        # Intended order of the leaf signature is the following:
        # leaf arrays, base scalars, transformation scalars.
        # So we are pre-filling scalars accumulator with base scalars before
        # stating depth-first walk.
        scalars = [name for name in base_names
            if name in self.nodes and self.nodes[name].type == NODE_SCALAR]
        visited = set(scalars)

        def visit(names):
            for name in names:
                if name in visited:
                    continue
                visited.add(name)

                # assuming that if we got a name not from the tree,
                # it is a temporary array
                if name not in self.nodes:
                    arrays.append(name)
                    continue

                node = self.nodes[name]
                if node.children is None:
                    if node.type == NODE_SCALAR:
                        scalars.append(name)
                    else:
                        arrays.append(name)
                else:
                    visit(node.children)

        visit(base_names)

        return [(name, self.nodes[name].value if name in self.nodes else None)
            for name in arrays + scalars]

    def base_values(self):
        return [self.nodes[name].value for name in self.base_names]

    def all_children(self, name):
        return [name for name, _ in self.leaf_signature([name])]

    def _clear_values(self):
        for name in self.nodes:
            self.nodes[name].value.clear()

    def set_temp_nodes(self, values_dict):
         self.temp_nodes = {name:AttrDict(value=value) for name, value in values_dict.items()}

    def propagate_to_base(self, values_dict):
        # takes {name: mock_val} and propagates it from leaves to roots,
        # updating nodes

        self._clear_values()

        def deduce(name):
            node = self.nodes[name]
            if node.children is None:
                # Values received from user may point to the same object.
                # Therefore we're playing it safe and not assigning them.
                node.value.fill_with(values_dict[name])
                return

            for child in node.children:
                deduce(child)

            # derive type
            child_dtypes = [self.nodes[child].value.dtype for child in node.children]
            tr = node.tr_to_children
            derive_types = tr.derive_i_from_os if node.type == NODE_OUTPUT else tr.derive_o_from_is
            node.value.dtype = dtypes.normalize_type(derive_types(*child_dtypes)[0])

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
            derive_types = tr.derive_os_from_i if node.type == NODE_OUTPUT else tr.derive_is_from_o
            arr_dtypes, scalar_dtypes = derive_types(node.value.dtype)
            arr_dtypes = dtypes.normalize_types(arr_dtypes)
            scalar_dtypes = dtypes.normalize_types(scalar_dtypes)
            for child, dtype in zip(node.children, arr_dtypes + scalar_dtypes):
                child_value = self.nodes[child].value
                if child_value.dtype is None:
                    child_value.dtype = dtype
                elif child_value.dtype != dtype:
                    raise TypePropagationError("Data type conflict in node " + child +
                        " while propagating types to leaves")

                # currently there is no shape derivation in transformations,
                # so we can just propagate it without checks
                if isinstance(child_value, ArrayValue):
                    child_value.shape = node.value.shape

                propagate(child)

        for name in self.base_names:
            # Values received from user may point to the same object.
            # Therefore we're playing it safe and not assigning them.
            self.nodes[name].value.fill_with(values_dict[name])
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
                if argname in self.nodes:
                    value = self.nodes[argname].value
                else:
                    value = self.temp_nodes[argname].value

                dtype = value.dtype
                ctype = dtypes.ctype(dtype)

                res.append(("GLOBAL_MEM " if value.is_array else " ") +
                    ctype + (" *" if value.is_array else " ") + leaf_name(argname))

            return ", ".join(res)

        def signature_macro(argnames):
            return "#define {macro_name} {arglist}".format(
                macro_name=signature_macro_name(),
                arglist=build_arglist(argnames))

        def process(name):
            if name in visited: return

            visited.add(name)
            node = self.nodes[name]

            if node.type == NODE_INPUT:
                if name in self.base_names:
                    leaf_macro = base_leaf_load_macro
                    node_macro = base_node_load_macro
                else:
                    leaf_macro = leaf_load_macro
                    node_macro = node_load_macro
            elif node.type == NODE_OUTPUT:
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

            if node.type == NODE_INPUT:
                definition = "INLINE WITHIN_KERNEL {outtype} {fname}({arglist}, int idx)".format(
                    outtype=dtypes.ctype(node.value.dtype),
                    fname=load_function_name(node.name),
                    arglist=build_arglist(all_children))
                input_names = node.children[:tr.inputs]
                scalar_names = node.children[tr.inputs:]

                args = {}
                for i, name in enumerate(input_names):
                    arg = TransformationArgument(self.nodes, 'i', i, name, node.type)
                    args[arg.label] = arg
                for i, name in enumerate(scalar_names):
                    arg = TransformationArgument(self.nodes, 's', i, name, node.type)
                    args[arg.label] = arg

                arg = TransformationArgument(self.nodes, 'o', 0, node.name, node.type)
                args[arg.label] = arg

            else:
                definition = "INLINE WITHIN_KERNEL void {fname}({arglist}, int idx, {intype} val)".format(
                    intype=dtypes.ctype(node.value.dtype),
                    fname=store_function_name(node.name),
                    arglist=build_arglist(all_children))

                output_names = node.children[:tr.outputs]
                scalar_names = node.children[tr.outputs:]

                args = {}
                for i, name in enumerate(output_names):
                    arg = TransformationArgument(self.nodes, 'o', i, name, node.type)
                    args[arg.label] = arg
                for i, name in enumerate(scalar_names):
                    arg = TransformationArgument(self.nodes, 's', i, name, node.type)
                    args[arg.label] = arg

                arg = TransformationArgument(self.nodes, 'i', 0, node.name, node.type)
                args[arg.label] = arg


            code_src = render_without_funcs(tr.code, func_c, **args)

            code_list.append("// node " + node.name + "\n" +
                definition + "\n{\n" + code_src + "\n}\n" +
                node_macro(node.name, all_children))

        for name in names:
            if name in self.base_names:
                process(name)
            else:
                code_list.append(base_leaf_load_macro(name))
                code_list.append(base_leaf_store_macro(name))

        leaf_names = [name for name, _ in self.leaf_signature(names)]
        return func_c.render() + "\n\n" + "\n\n".join(code_list) + \
            "\n\n" + signature_macro(leaf_names)

    def connections_for(self, names):
        connections = []

        def visit(name):
            node = self.nodes[name]
            children = node.children
            if children is None:
                return
            array_children = [n for n in children if self.nodes[n].value.is_array]
            scalar_children = [n for n in children if not self.nodes[n].value.is_array]
            connections.append((node.tr_to_children, name, array_children, scalar_children))
            for n in array_children:
                visit(n)

        for name in names:
            if name not in self.temp_nodes:
                visit(name)

        return connections

    def has_array_leaf(self, name):
        names = set(n for n, v in self.leaf_signature() if v.is_array)
        return name in names

    def connect(self, tr, array_arg, new_array_args, new_scalar_args):

        if not self.has_array_leaf(array_arg):
            raise ValueError("Argument " + array_arg +
                " does not exist or is not suitable for connection")

        for name in new_array_args + new_scalar_args:
            if not valid_argument_name(name):
                raise ValueError("Incorrect argument name: " + name)

        parent = self.nodes[array_arg]

        if parent.type == NODE_OUTPUT:
            if tr.inputs > 1:
                raise ValueError("Transformation for an output node must have one input")
            if tr.outputs != len(new_array_args):
                raise ValueError("Number of array argument names does not match the transformation")

        if parent.type == NODE_INPUT:
            if tr.outputs > 1:
                raise ValueError("Transformation for an input node must have one output")
            if tr.inputs != len(new_array_args):
                raise ValueError("Number of array argument names does not match the transformation")

        if tr.scalars != len(new_scalar_args):
            raise ValueError("Number of scalar argument names does not match the transformation")

        # Delay applying changes until the end of the method,
        # in case we get an error in the process.
        new_nodes = {}

        for name in new_array_args:
            if name in self.nodes:
                if self.nodes[name].type == NODE_SCALAR:
                    raise ValueError("Argument " + name + " is a scalar, expected an array")
                if parent.type == NODE_OUTPUT:
                    raise ValueError("Cannot connect to an existing output node")
            else:
                new_nodes[name] = AttrDict(
                    name=name, type=parent.type,
                    value=ArrayValue(None, None),
                    children=None, tr_to_children=None)
        for name in new_scalar_args:
            if name in self.nodes:
                if self.nodes[name].type != NODE_SCALAR:
                    raise ValueError("Argument " + name + " is an array, expected a scalar")
            else:
                new_nodes[name] = AttrDict(
                    name=name, type=NODE_SCALAR,
                    value=ScalarValue(None, None),
                    children=None, tr_to_children=None)

        parent.children = new_array_args + new_scalar_args
        parent.tr_to_children = tr
        self.nodes.update(new_nodes)
