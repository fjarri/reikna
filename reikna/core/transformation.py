import weakref

from reikna.helpers import template_for, template_def, Graph
import reikna.cluda.dtypes as dtypes
from reikna.cluda import Module, Snippet
from reikna.core.signature import Signature, Type, Parameter


TEMPLATE = template_for(__file__)


class TransformationParameter(Type):
    """
    Bases: :py:class:`~reikna.core.Type`

    Represents a typed transformation parameter.
    Can be used as a substitute of an array for functions
    which are only interested in array metadata.

    .. py:attribute:: name

        Parameter name.
    """

    def __init__(self, tr, name, type_):
        Type.__init__(self, type_.dtype, shape=type_.shape, strides=type_.strides)
        self._tr = weakref.ref(tr)
        self.name = name

    def belongs_to(self, tr):
        return self._tr() is tr


class Transformation:
    """
    A class containing a pure parallel transformation of arrays.

    A transformation:

    * cannot use local memory;
    * cannot use global/local id getters (and depends only on externally passed indices);
    * cannot have 'io' arguments;
    * has at least one argument that uses
      :py:attr:`~reikna.core.transformation.KernelParameter.load_same` or
      :py:attr:`~reikna.core.transformation.KernelParameter.store_same`, and does it only once.

    :param parameters: a list of :py:class:`~reikna.core.Parameter` objects.
    :param code: a source template for the transformation.
        Will be wrapped in a template def with positional arguments with the names of
        objects in ``parameters``.
    :param render_kwds: a dictionary with render keywords that will be passed to the snippet.
    :param connectors: a list of parameter names suitable for connection.
    """
    def __init__(self, parameters, code, render_kwds=None, connectors=None):

        self.signature = Signature(parameters)

        for param in self.signature.parameters.values():
            setattr(
                self, param.name,
                TransformationParameter(self, param.name, param.annotation.type))

        if connectors is not None:
            self.connectors = connectors
        else:
            self.connectors = [param.name for param in parameters if param.annotation.array]

        tr_param_names = ['idxs'] + [param.name for param in self.signature.parameters.values()]
        self.snippet = Snippet(template_def(tr_param_names, code), render_kwds=render_kwds)


class Node:

    def __init__(self, param, input_ntr=None, output_ntr=None):
        self.param = param
        self.input_ntr = input_ntr
        self.output_ntr = output_ntr

    def connect(self, ntr):
        if ntr.output:
            return Node(self.param, input_ntr=self.input_ntr, output_ntr=ntr)
        else:
            return Node(self.param, input_ntr=ntr, output_ntr=self.output_ntr)

    def get_child_names(self):
        get_names = lambda ntr: [] if ntr is None else ntr.get_child_names()
        return get_names(self.output_ntr) + get_names(self.input_ntr)

    def get_connections(self):
        get_conn = lambda ntr: [] if ntr is None else [ntr]
        return get_conn(self.output_ntr) + get_conn(self.input_ntr)


class NodeTransformation:

    def __init__(self, connector_node_name, tr, param_connections, output=False):
        self.tr = tr
        self.connector_node_name = connector_node_name
        self.output = output
        self.node_from_tr = {
            tr_param_name:param_name for param_name, tr_param_name in param_connections.items()}

    def get_child_names(self):
        names = []
        # Walking the tree conserving the order of parameters in the transformation.
        for tr_param in self.tr.signature.parameters.values():
            node_name = self.node_from_tr[tr_param.name]
            if node_name != self.connector_node_name:
                names.append(node_name)
        return names

    def translate_node_names(self, translator):
        tr_names, node_names = zip(*list(self.node_from_tr.items()))
        return NodeTransformation(
            translator(self.connector_node_name),
            self.tr,
            list(map(translator, node_names)),
            tr_names,
            output=self.output)


class TransformationTree:

    def __init__(self, root_params):
        self.root_signature = Signature(root_params)
        self.nodes = {param.name:Node(param) for param in root_params}

    def _get_subtree_names(self, names, visited, leaves_only=False):

        result = []

        for name in names:
            if name in visited:
                continue
            visited.add(name)

            child_names = self.nodes[name].get_child_names()

            if not leaves_only or len(child_names) == 0:
                result.append(name)

            result += self._get_subtree_names(
                child_names, visited, leaves_only=leaves_only)

        return result

    def get_subtree_names(self, root_names, leaves_only=False):
        return self._get_subtree_names(root_names, set(), leaves_only=leaves_only)

    def get_leaf_signature(self):
        root_names = [param.name for param in self.root_signature.parameters.values()]
        leaf_names = self.get_subtree_names(root_names, leaves_only=True)
        return Signature([self.nodes[name].param for name in leaf_names])

    def get_node_parameters(self):
        root_names = [param.name for param in self.root_signature.parameters.values()]
        node_names = self.get_subtree_names(root_names, leaves_only=False)
        return [self.nodes[name].param for name in node_names]

    def _connect(self, ntr):

        # New nodes introduced by the transformation
        new_nodes = {}
        for tr_param in ntr.tr.signature.parameters.values():
            node_name = ntr.node_from_tr[tr_param.name]
            if node_name not in self.nodes:
                new_param = Parameter(node_name, tr_param.annotation)
                new_nodes[node_name] = Node(new_param)

        old_node = self.nodes[ntr.connector_node_name]
        new_nodes[ntr.connector_node_name] = old_node.connect(ntr)

        # Delaying the internal changes before all the data structures are created.
        # This way a failed .connect() does not break any internals
        # (mostly important for interactive regime).
        self.nodes.update(new_nodes)

    def connect(self, param_name, tr, param_connections):

        # Check:
        # - there's a node ``leaf_name`` and it doesn't have a connection of this type attached
        # - check that ``primary_conn`` exist in ``tr``
        # - check that keys of ``connections.from_tr`` point to nodes of proper types, or are new names
        # - check that values of ``connections.from_node`` exist in ``tr``
        output = self.nodes[param_name].param.annotation.output
        self._connect(NodeTransformation(param_name, tr, param_connections, output=output))

    def reconnect(self, other_tree, translator=None):
        for ntr in other_tree.connections():
            if translator is not None:
                ntr = ntr.translate_node_names(translator)
            if ntr.connector_node_name in self.nodes:
                self._connect(ntr)

    def connections(self):
        root_names = [param.name for param in self.root_signature.parameters.values()]
        node_names = self.get_subtree_names(root_names, leaves_only=False)
        for name in node_names:
            node = self.nodes[name]
            for ntr in node.get_connections():
                yield ntr

    def translate(self, translator):
        root_params = self.root_signature.parameters.values()
        new_root_params = [param.rename(translator(param.name)) for param in root_params]
        new_tree = TransformationTree(new_root_params)
        new_tree.reconnect(self, translator=translator)

        return new_tree

    def get_subtree(self, argnames, parameters):
        new_params = [
            self.nodes[name].param if name in self.nodes else parameters[name]
            for name in argnames]
        new_tree = TransformationTree(new_params)
        new_tree.reconnect(self)
        return new_tree

    def get_kernel_definition(self, kernel_name):
        leaf_params = self.get_leaf_signature().parameters.values()
        return TEMPLATE.get_def('kernel_definition').render(
            kernel_name, leaf_params, leaf_name=leaf_name)

    def _get_transformation_module(self, ntr):

        # HACK: Technically, ``module`` attribute is not documented.
        # The reason it is used here is that I need to keep generation of C names for
        # index variable in one place, and the template is the best choice
        # (because they are mostly used there).
        param = self.nodes[ntr.connector_node_name].param
        index_cnames = TEMPLATE.module.index_cnames(param)

        if ntr.output:
            connector_def = "node_output_connector"
            transformation_def = "node_output_transformation"
        else:
            connector_def = "node_input_connector"
            transformation_def = "node_input_transformation"

        tr_args = [index_cnames]
        connection_names = []
        for tr_param in ntr.tr.signature.parameters.values():
            connection_name = ntr.node_from_tr[tr_param.name]
            connection_names.append(connection_name)

            if connection_name == ntr.connector_node_name:
                if ntr.output:
                    load_same = TEMPLATE.get_def(connector_def).render()
                    tr_args.append(KernelParameter(
                        param.name, param.annotation.type, load_same=load_same))
                else:
                    store_same = TEMPLATE.get_def(connector_def).render()
                    tr_args.append(KernelParameter(
                        param.name, param.annotation.type, store_same=store_same))
            else:
                tr_args.append(self._get_argobject(connection_name))

        subtree_names = self.get_subtree_names([ntr.connector_node_name], leaves_only=True)
        subtree_params = [self.nodes[name].param for name in subtree_names]

        return Module(
            TEMPLATE.get_def(transformation_def),
            render_kwds=dict(
                tr_snippet=ntr.tr.snippet,
                tr_args=tr_args,
                param=param,
                subtree_params=subtree_params,
                leaf_name=leaf_name))

    def _get_argobject(self, name, base=False):
        # Takes a base argument name and returns the corresponding Argument object
        # which can be passed to the main kernel.
        # If the name is not in base, it is treated as a leaf.

        node = self.nodes[name]
        param = node.param

        if not param.annotation.array:
            return KernelParameter(param.name, param.annotation.type)

        load_idx = None
        store_idx = None
        load_same = None
        store_same = None
        load_combined_idx = None
        store_combined_idx = None

        if param.annotation.input:
            if node.input_ntr is None:
                load_idx = Module(
                    TEMPLATE.get_def('leaf_input_macro'),
                    render_kwds=dict(param=param, leaf_name=leaf_name))
            else:
                load_idx = self._get_transformation_module(node.input_ntr)

            subtree_names = self.get_subtree_names([name], leaves_only=True)
            subtree_params = [self.nodes[st_name].param for st_name in subtree_names]

            if not base:
                load_same = Module(
                    TEMPLATE.get_def('node_input_same_indices'),
                    render_kwds=dict(
                        param=param, load_idx=load_idx, leaf_name=leaf_name,
                        subtree_params=subtree_params))

            load_combined_idx = Module(
                TEMPLATE.get_def('node_input_combined'),
                render_kwds=dict(param=param, leaf_name=leaf_name, load_idx=load_idx,
                    subtree_params=subtree_params))

        if param.annotation.output:
            if node.output_ntr is None:
                store_idx = Module(
                    TEMPLATE.get_def('leaf_output_macro'),
                    render_kwds=dict(param=param, leaf_name=leaf_name))
            else:
                store_idx = self._get_transformation_module(node.output_ntr)

            subtree_names = self.get_subtree_names([name], leaves_only=True)
            subtree_params = [self.nodes[st_name].param for st_name in subtree_names]

            if not base:
                store_same = Module(
                    TEMPLATE.get_def('node_output_same_indices'),
                    render_kwds=dict(
                        param=param, store_idx=store_idx, leaf_name=leaf_name,
                        subtree_params=subtree_params))

            store_combined_idx = Module(
                TEMPLATE.get_def('node_output_combined'),
                render_kwds=dict(param=param, leaf_name=leaf_name, store_idx=store_idx,
                    subtree_params=subtree_params))

        return KernelParameter(
            param.name, param.annotation.type,
            load_idx=load_idx,
            store_idx=store_idx,
            load_same=load_same,
            store_same=store_same,
            load_combined_idx=load_combined_idx,
            store_combined_idx=store_combined_idx)

    def get_argobjects(self):
        return [self._get_argobject(param.name, base=True)
            for param in self.root_signature.parameters.values()]


def leaf_name(name):
    return "_leaf_" + name


class KernelParameter(Type):
    """
    Bases: :py:class:`~reikna.core.Type`

    Providing an interface for accessing kernel arguments in a template.
    Depending on the parameter type, and whether it is used
    inside a computation or a transformation template,
    can have different load/store attributes available.

    .. py:attribute:: name

        Parameter name

    .. py:method:: __str__()

        Returns the C kernel parameter name corresponding to this parameter.
        It is the only method available for scalar parameters.

    .. py:attribute:: load_idx

        A module providing a macro with the signature ``(idx0, idx1, ...)``,
        returning the corresponding element of the array.

    .. py:attribute:: store_idx

        A module providing a macro with the signature ``(idx0, idx1, ..., val)``,
        saving ``val`` into the specified position.

    .. py:method:: load_combined_idx(slices)

        A module providing a macro with the signature ``(cidx0, cidx1, ...)``,
        returning the element of the array corresponding to the new slicing of indices
        (e.g. an array with shape ``(2, 3, 4, 5, 6)`` sliced as ``slices=(2, 2, 1)``
        is indexed as an array with shape ``(6, 20, 6)``).

    .. py:method:: store_combined_idx(slices)

        A module providing a macro with the signature ``(cidx0, cidx1, ..., val)``,
        saving ``val`` into the specified position
        corresponding to the new slicing of indices.

    .. py:attribute:: load_same

        A module providing a macro that returns the element of the array
        corresponding to the indices used by the caller of the transformation.

    .. py:attribute:: store_same

        A module providing a macro with the signature ``(val)`` that stores ``val``
        using the indices used by the caller of the transformation.
    """

    def __init__(self, name, type_, load_idx=None, store_idx=None, load_same=None, store_same=None,
            load_combined_idx=None, store_combined_idx=None):
        """__init__()""" # hide the signature from Sphinx

        Type.__init__(self, type_.dtype, shape=type_.shape, strides=type_.strides)

        self._leaf_name = leaf_name(name)
        self.name = name

        if load_idx is not None: self.load_idx = load_idx
        if store_idx is not None: self.store_idx = store_idx
        if load_same is not None: self.load_same = load_same
        if store_same is not None: self.store_same = store_same
        if load_combined_idx is not None: self.load_combined_idx = load_combined_idx
        if store_combined_idx is not None: self.store_combined_idx = store_combined_idx

    def __process_modules__(self, process):
        kwds = {}
        attrs = (
            'load_idx', 'store_idx', 'load_same', 'store_same',
            'load_combined_idx', 'store_combined_idx')
        for attr in attrs:
            if hasattr(self, attr):
                kwds[attr] = process(getattr(self, attr))

        return KernelParameter(self.name, self, **kwds)

    def __repr__(self):
        return "KernelParameter("+ self.name + ")"

    def __str__(self):
        return self._leaf_name
