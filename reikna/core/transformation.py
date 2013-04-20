import re

import numpy

import reikna.cluda.dtypes as dtypes
from reikna.cluda import Module
from reikna.helpers import AttrDict, product, wrap_in_tuple, template_func, template_for


TEMPLATE = template_for(__file__)


class TypePropagationError(Exception):
    pass


def valid_argument_name(name):
    return (re.match(r"^[a-zA-Z_]\w*$", name) is not None)


class ArrayValue(object):
    """
    Stub object for array arguments.

    .. py:attribute:: shape

        Tuple specifying the shape of the array.

    .. py:attribute:: dtype

        :py:class:`numpy.dtype` object specifying the data type of the array.
    """

    def __init__(self, shape, dtype):
        self.shape = wrap_in_tuple(shape) if shape is not None else None
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
    """
    Stub object for scalar arguments.

    .. py:attribute:: dtype

        :py:class:`numpy.dtype` object specifying the data type of the scalar.
    """

    def __init__(self, dtype):
        self.dtype = dtypes.normalize_type(dtype) if dtype is not None else None
        self.is_array = False

    def fill_with(self, other):
        self.dtype = other.dtype

    def clear(self):
        self.dtype = None

    def __str__(self):
        props = ["scalar"]
        if self.dtype is not None:
            props.append(str(self.dtype))
        return ", ".join(props)

    def __repr__(self):
        return "ScalarValue(" + repr(self.dtype) + ")"


def wrap_value(value):
    if isinstance(value, ScalarValue) or isinstance(value, ArrayValue):
        return value
    elif hasattr(value, 'dtype'):
        if hasattr(value, 'shape') and len(value.shape) > 0:
            return ArrayValue(value.shape, value.dtype)
        else:
            return ScalarValue(value.dtype)
    else:
        dtype = dtypes.min_scalar_type(value)
        return ScalarValue(dtype)


class Transformation:
    """
    Defines an elementwise transformation.

    :param inputs: list with input value names (these names will be used in the template),
        or the number of input array values (in which case they will be given names
        ``i1``, ``i2`` etc).
    :param outputs: list with output value names (these names will be used in the template),
        or the number of input array values (in which case they will be given names
        ``o1``, ``o2`` etc).
    :param scalars: list with scalar parameter names (these names will be used in the template),
        or the number of input array values (in which case they will be given names
        ``p1``, ``p2`` etc).
    :param derive_o_from_is: a function taking dtypes of ``inputs`` and ``scalars``,
        and returning the output dtype.
        If ``None``, :py:func:`~reikna.cluda.dtypes.result_type` is used.
        Called when the transformation is connected to the input argument.
    :param derive_i_from_os: a function taking dtypes of ``outputs`` and ``scalars``,
        and returning the input dtype.
        If ``None``, :py:func:`~reikna.cluda.dtypes.result_type` is used.
        Called when the transformation is connected to the output argument.
    :param code: template source with the transformation code.
        See :ref:`tutorial-advanced-transformation` section for details.
    """

    def __init__(self, inputs=1, outputs=1, scalars=0,
            derive_o_from_is=None, derive_i_from_os=None,
            derive_render_kwds=None,
            snippet=None):

        gen_names = lambda names, prefix: [prefix + str(i+1) for i in xrange(names)] \
            if isinstance(names, int) else list(names)
        self.inputs = gen_names(inputs, 'i')
        self.outputs = gen_names(outputs, 'o')
        self.scalars = gen_names(scalars, 's')

        if snippet is None:
            snippet = "${" + self.outputs[0] + ".store}(${" + self.inputs[0] + ".load});"

        self.snippet_template = template_func(
            self.outputs + self.inputs + self.scalars,
            snippet)

        if derive_o_from_is is None:
            if len(self.outputs) == 1:
                derive_o_from_is = dtypes.result_type
        else:
            if len(self.outputs) > 1:
                raise ValueError(
                    "This transformation cannot be used for an input and therefore cannot have "
                    "a ``derive_o_from_is`` parameter")

        if derive_i_from_os is None:
            if len(self.inputs) == 1:
                derive_i_from_os = dtypes.result_type
        else:
            if len(self.inputs) > 1:
                raise ValueError(
                    "This transformation cannot be used for an output and therefore cannot have "
                    "a ``derive_i_from_os`` parameter")

        if derive_render_kwds is None:
            derive_render_kwds = lambda *args: {}

        self.derive_o_from_is = derive_o_from_is
        self.derive_i_from_os = derive_i_from_os
        self.derive_render_kwds = derive_render_kwds

    def construct_snippet(self, *args):
        render_kwds = self.derive_render_kwds(*args)
        return Module(self.snippet_template, render_kwds=render_kwds, snippet=True)


class Node:

    INPUT = "input node"
    OUTPUT = "output node"
    SCALAR = "scalar node"
    TEMP = "temporary node"

    def __init__(self, name, node_type, value=None):
        self.name = name
        self.leaf_name = "_leaf_" + name
        self.type = node_type
        if value is None:
            value = ScalarValue(None) if node_type == self.SCALAR else ArrayValue(None, None)
        self.value = value
        self.children=None
        self.tr_to_children=None

    def __repr__(self):
        return repr((self.type, self.name))


class TransformationArgument(AttrDict):

    def __init__(self, node, load=None, store=None):
        AttrDict.__init__(self)
        self._node = node

        self.dtype = node.value.dtype
        self.ctype = dtypes.ctype(self.dtype)

        if load is not None:
            self.load = load
        if store is not None:
            self.store = store


class ConnectorArgument:

    def __init__(self, node):
        self.dtype = node.value.dtype
        self.ctype = dtypes.ctype(self.dtype)

        connector = TEMPLATE.get_def('connector').render(node)

        if node.type == Node.INPUT:
            self.store = connector
        else:
            self.load = connector


class ScalarArgument:

    def __init__(self, node):
        self.leaf_name = node.leaf_name
        self.dtype = node.value.dtype
        self.ctype = dtypes.ctype(self.dtype)

    def __str__(self):
        return self.leaf_name


class TransformationTree:

    def __init__(self, outputs, inputs, scalars):
        self._outputs = outputs
        self._inputs = inputs
        self._scalars = scalars

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
            self.nodes[name] = Node(name, Node.OUTPUT)
        for name in inputs:
            self.nodes[name] = Node(name, Node.INPUT)
        for name in scalars:
            self.nodes[name] = Node(name, Node.SCALAR)

    def copy(self):
        tree = TransformationTree(self._outputs, self._inputs, self._scalars)

        # recreate connections
        connections = self.connections_for(self._outputs + self._inputs + self._scalars)
        for tr, array_arg, new_array_args, new_scalar_args in connections:
            tree.connect(tr, array_arg, new_array_args, new_scalar_args)

        # repopulate nodes
        tree.propagate_to_base({name:value for name, value in self.leaf_signature()})

        return tree

    def leaf_signature(self, base_names=None):

        if base_names is None:
            base_names = self.base_names

        arrays = []

        # Intended order of the leaf signature is the following:
        # leaf arrays, base scalars, transformation scalars.
        # So we are pre-filling scalars accumulator with base scalars before
        # stating depth-first walk.
        scalars = [name for name in base_names
            if name in self.nodes and self.nodes[name].type == Node.SCALAR]
        visited = set(scalars)

        def visit(names):
            for name in names:
                if name in visited:
                    continue
                visited.add(name)

                node = self.nodes[name]
                if node.children is None:
                    if node.type == Node.SCALAR:
                        scalars.append(name)
                    else:
                        arrays.append(name)
                else:
                    visit(node.children)

        visit(base_names)

        return [(name, self.nodes[name].value) for name in arrays + scalars]

    def base_values(self):
        return [self.nodes[name].value for name in self.base_names]

    def leaf_values_dict(self, base_names=None):
        return {name:value for name, value in self.leaf_signature(base_names=base_names)}

    def all_children(self, name):
        return [name for name, _ in self.leaf_signature([name])]

    def add_temp_node(self, name, value):
         self.nodes[name] = Node(name, Node.TEMP, value=value)

    def propagate_to_base(self, values_dict):
        # takes {name: mock_val} and propagates it from leaves to roots,
        # updating nodes

        # clear the transformation tree
        for name in self.nodes:
            self.nodes[name].value.clear()

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
            derive_types = tr.derive_i_from_os if node.type == Node.OUTPUT else tr.derive_o_from_is
            node.value.dtype = dtypes.normalize_type(derive_types(*child_dtypes))

            # derive shape
            child_shapes = [self.nodes[child].value.shape for child in node.children
                if hasattr(self.nodes[child].value, 'shape')]
            assert len(set(child_shapes)) == 1
            node.value.shape = child_shapes[0]

        for name in self.base_names:
            deduce(name)

    def _transformations_for(self, name, base=False):
        # Takes a base argument name and returns the corresponding Argument object
        # which can be passed to the main kernel.
        # If the name is not in base, it is treated as a leaf.

        node = self.nodes[name]

        if node.type == Node.SCALAR:
            return ScalarArgument(node)

        if node.type == Node.TEMP:
            module_load = Module(
                TEMPLATE.get_def('leaf_macro'),
                render_kwds=dict(node=node, node_type=node.INPUT, base=base))
            module_store = Module(
                TEMPLATE.get_def('leaf_macro'),
                render_kwds=dict(node=node, node_type=node.OUTPUT, base=base))
            return TransformationArgument(node, load=module_load, store=module_store)

        if node.children is None:
            module = Module(
                TEMPLATE.get_def('leaf_macro'),
                render_kwds=dict(node=node, node_type=node.type, base=base))
            if node.type == Node.INPUT:
                return TransformationArgument(node, load=module)
            else:
                return TransformationArgument(node, store=module)

        tr = node.tr_to_children
        if node.type == Node.INPUT:
            tr_args = (
                [ConnectorArgument(node)] +
                [self._transformations_for(name) for name in node.children])
            tr_names = [node.name] + node.children
        else:
            outputs = node.children[:len(tr.outputs)]
            scalars = node.children[len(tr.outputs):]
            tr_args = (
                [self._transformations_for(name) for name in outputs] +
                [ConnectorArgument(node)] +
                [self._transformations_for(name) for name in scalars])
            tr_names = outputs + [node.name] + scalars

        all_children = self.all_children(node.name)

        tr_dtypes = [self.nodes[name].value.dtype for name in tr_names]
        tr_snippet = tr.construct_snippet(*tr_dtypes)

        render_kwds=dict(
            tr_snippet=tr_snippet,
            tr_args=tr_args,
            node=node,
            base=base,
            leaf_nodes=[self.nodes[name] for name in all_children])

        module = Module(
            TEMPLATE.get_def('transformation_node'),
            render_kwds=render_kwds)

        if node.type == Node.INPUT:
            return TransformationArgument(node, load=module)
        else:
            return TransformationArgument(node, store=module)

    def transformations_for(self, kernel_name, names):
        # Takes [name] for bases and returns a list with Argument objects
        # corresponding to the list of names.
        # If some of the names are not in base, they are treated as leaves.

        leaf_nodes = [self.nodes[name] for name, _ in self.leaf_signature(names)]

        kernel_def = TEMPLATE.get_def('kernel_definition').render(
            kernel_name, leaf_nodes, dtypes=dtypes)
        tr_args = [self._transformations_for(name, base=True) for name in names]

        return kernel_def, tr_args

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

        if parent.type == Node.OUTPUT:
            if len(tr.inputs) > 1:
                raise ValueError("Transformation for an output node must have one input")
            if len(tr.outputs) != len(new_array_args):
                raise ValueError("Number of array argument names does not match the transformation")

        if parent.type == Node.INPUT:
            if len(tr.outputs) > 1:
                raise ValueError("Transformation for an input node must have one output")
            if len(tr.inputs) != len(new_array_args):
                raise ValueError("Number of array argument names does not match the transformation")

        if len(tr.scalars) != len(new_scalar_args):
            raise ValueError("Number of scalar argument names does not match the transformation")

        # Delay applying changes until the end of the method,
        # in case we get an error in the process.
        new_nodes = {}

        for name in new_array_args:
            if name in self.nodes:
                if self.nodes[name].type == Node.SCALAR:
                    raise ValueError("Argument " + name + " is a scalar, expected an array")
                if parent.type == Node.OUTPUT:
                    raise ValueError("Cannot connect to an existing output node")
            else:
                new_nodes[name] = Node(name, parent.type)
        for name in new_scalar_args:
            if name in self.nodes:
                if self.nodes[name].type != Node.SCALAR:
                    raise ValueError("Argument " + name + " is an array, expected a scalar")
            else:
                new_nodes[name] = Node(name, Node.SCALAR)

        parent.children = new_array_args + new_scalar_args
        parent.tr_to_children = tr
        self.nodes.update(new_nodes)
