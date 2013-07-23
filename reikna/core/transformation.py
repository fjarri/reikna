import weakref

from reikna.helpers import template_for, template_def, Graph
import reikna.cluda.dtypes as dtypes
from reikna.cluda import Module, Snippet
from reikna.core.signature import Signature, Type, Parameter, Annotation


TEMPLATE = template_for(__file__)


class TransformationParameter(Type):
    """
    Bases: :py:class:`~reikna.core.Type`

    Represents a typed transformation parameter.
    Can be used as a substitute of an array for functions
    which are only interested in array metadata.
    """

    def __init__(self, tr, name, type_):
        Type.__init__(self, type_.dtype, shape=type_.shape, strides=type_.strides)
        self._tr = weakref.ref(tr)
        self._name = name

    def belongs_to(self, tr):
        return self._tr() is tr

    def __str__(self):
        return self._name


class Transformation:
    """
    A class containing a pure parallel transformation of arrays.
    Some restrictions apply:

    * it cannot use local memory;
    * it cannot use global/local id getters (and depends only on externally passed indices);
    * it cannot have ``'io'`` arguments;
    * it has at least one argument that uses
      :py:attr:`~reikna.core.transformation.KernelParameter.load_same` or
      :py:attr:`~reikna.core.transformation.KernelParameter.store_same`, and does it only once.

    :param parameters: a list of :py:class:`~reikna.core.Parameter` objects.
    :param code: a source template for the transformation.
        Will be wrapped in a template def with positional arguments with the names of
        objects in ``parameters``.
    :param render_kwds: a dictionary with render keywords that will be passed to the snippet.
    :param connectors: a list of parameter names suitable for connection.
        Defaults to all non-scalar parameters.
    """
    def __init__(self, parameters, code, render_kwds=None, connectors=None):

        for param in parameters:
            if param.annotation.input and param.annotation.output:
                raise ValueError(
                    "Transformation cannot have 'io' parameters ('" + param.name + "')")

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
    """
    Represents a node in the transformation tree.
    """

    def __init__(self, input_ntr=None, output_ntr=None):
        self.input_ntr = input_ntr
        self.output_ntr = output_ntr

    def connect(self, ntr):
        if ntr.output:
            return Node(input_ntr=self.input_ntr, output_ntr=ntr)
        else:
            return Node(input_ntr=ntr, output_ntr=self.output_ntr)

    def get_child_names(self):
        get_names = lambda ntr: [] if ntr is None else ntr.get_child_names()
        return get_names(self.output_ntr) + get_names(self.input_ntr)

    def get_connections(self):
        get_conn = lambda ntr: [] if ntr is None else [ntr]
        return get_conn(self.output_ntr) + get_conn(self.input_ntr)


class NodeTransformation:
    """
    Represents a transformation between two nodes
    (an edge of the transformation tree).
    """

    def __init__(self, connector_node_name, tr, param_connections):
        self.tr = tr
        self.connector_node_name = connector_node_name
        self.node_from_tr = {
            tr_param_name:param_name for param_name, tr_param_name in param_connections.items()}

        # Check that all transformation parameter names are represented,
        # and each of them is only mentioned once.
        tr_names_given = set(self.node_from_tr)
        tr_names_available = set(tr.signature.parameters)
        if tr_names_given != tr_names_available:
            raise ValueError(
                "Supplied transformation names {given} "
                " do not fully coincide with the existing ones: {available}".format(
                    given=tr_names_given, available=tr_names_available))

        # If node names were not guaranteed to be different
        # (which is enforced by Computation.connect()),
        # we would have to make the same check for them as well.

        # Taking into account that 'io' parameters are not allowed for transformations.
        tr_connector = param_connections[connector_node_name]
        self.output = tr.signature.parameters[tr_connector].annotation.input

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
    """
    A directed acyclic graph of transformations of the kernel arguments.
    """

    def __init__(self, root_parameters):
        # Preserve order of initial root parameters.
        # These can repeat.
        self.root_names = []

        self.root_annotations = {}

        self.nodes = {} # all nodes of the tree
        self.leaf_annotations = {} # nodes available for connection

        for param in root_parameters:
            self.root_names.append(param.name)
            if param.name in self.root_annotations:
                # safety check
                assert param.annotation == self.root_annotations[param.name]
            else:
                self.nodes[param.name] = Node()
                self.root_annotations[param.name] = param.annotation
                self.leaf_annotations[param.name] = param.annotation

    def _get_subtree_names(self, names, visited, leaves_only=False):
        """Helper method for traversing the tree."""

        result = []

        for name in names:
            if name in visited:
                continue
            visited.add(name)

            node = self.nodes[name]
            ann = self.leaf_annotations[name] if name in self.leaf_annotations else None
            child_names = node.get_child_names()

            subtree_names = self._get_subtree_names(child_names, visited, leaves_only=leaves_only)
            has_input_subtree = node.input_ntr is not None
            has_output_subtree = node.output_ntr is not None
            leaf_node = name in self.leaf_annotations

            name_present = False
            if not leaves_only or len(subtree_names) == 0 or (leaf_node and not has_output_subtree):
                result.append(name)
                name_present = True
            result += subtree_names
            if not name_present and leaf_node and not has_input_subtree:
                result.append(name)

        return result

    def get_subtree_names(self, root_names=None, leaves_only=False):
        if root_names is None:
            root_names = self.root_names
        return self._get_subtree_names(root_names, set(), leaves_only=leaves_only)

    def get_root_annotations(self):
        return self.root_annotations

    def get_root_parameters(self):
        return [Parameter(name, self.root_annotations[name]) for name in self.root_names]

    def get_leaf_parameters(self, root_names=None):
        leaf_names = self.get_subtree_names(root_names=root_names, leaves_only=True)
        return [Parameter(name, self.leaf_annotations[name]) for name in leaf_names]

    def _connect(self, ntr):

        # At this point we assume that ``ntr`` describes a valid connection.
        # All sanity checks are performed in ``connect()``.

        for tr_param in ntr.tr.signature.parameters.values():
            node_name = ntr.node_from_tr[tr_param.name]

            if node_name == ntr.connector_node_name:
                ann = self.leaf_annotations[node_name]
                if ann.input and ann.output:
                # splitting the 'io' leaf
                    updated_role = 'i' if ntr.output else 'o'
                    self.leaf_annotations[node_name] = Annotation(ann.type, role=updated_role)
                else:
                # 'i' or 'o' leaf is hidden by the transformation
                    del self.leaf_annotations[node_name]

            else:
                if node_name in self.leaf_annotations and self.leaf_annotations[node_name].array:
                    ann = self.leaf_annotations[node_name]
                    if (ann.input and ntr.output) or (ann.output and ntr.input):
                    # joining 'i' and 'o' paths into an 'io' leaf
                        self.leaf_annotations[node_name] = Annotation(ann.type, role='io')
                else:
                    self.leaf_annotations[node_name] = tr_param.annotation

            if node_name not in self.nodes:
                self.nodes[node_name] = Node()

        self.nodes[ntr.connector_node_name] = self.nodes[ntr.connector_node_name].connect(ntr)

    def connect(self, comp_connector, tr, param_connections):

        ntr = NodeTransformation(comp_connector, tr, param_connections)

        # Check that the target actually exists
        if comp_connector not in self.leaf_annotations:
            raise ValueError("Parameter '" + comp_connector + "' is not a part of the signature.")

        # Check that the types of connections are correct
        for node_name, tr_name in param_connections.items():
            if node_name not in self.leaf_annotations:
                if node_name == comp_connector:
                    raise ValueError(
                        "Parameter '" + node_name + "' is not a part of the signature")
                elif node_name in self.nodes:
                    raise ValueError(
                        "Parameter '" + node_name + "' is hidden by transformations")

            if node_name not in self.leaf_annotations:
                # If node names could repeat, we would have to check that
                # transformation parameters with incompatible types are not pointing
                # at the same new node.
                continue

            node_ann = self.leaf_annotations[node_name]
            tr_ann = tr.signature.parameters[tr_name].annotation

            if tr_ann.type != node_ann.type:
                raise ValueError(
                    "Incompatible types of the transformation parameter '{tr_name}' ({tr_type}) "
                    "and the node '{node_name}' ({node_type})".format(
                        node_name=node_name, tr_name=tr_name,
                        node_type=node_ann.type, tr_type=tr_ann.type))

            # No more to check in the case of scalars
            if not tr_ann.array:
                continue

            if node_name == comp_connector:
                if ntr.output and not node_ann.output:
                    raise ValueError("'" + node_name + "' is not an output node")
                if not ntr.output and not node_ann.input:
                    raise ValueError("'" + node_name + "' is not an input node")
            else:
                if ntr.output and node_ann.output:
                    raise ValueError(
                        "Cannot connect transformation parameter '{tr_name}' "
                        "to an existing output node '{node_name}'".format(
                            tr_name=tr_name, node_name=node_name))

        self._connect(ntr)

    def reconnect(self, other_tree, translator=None):
        for ntr in other_tree.connections():
            if translator is not None:
                ntr = ntr.translate_node_names(translator)
            if ntr.connector_node_name in self.nodes:
                self._connect(ntr)

    def connections(self):
        node_names = self.get_subtree_names(leaves_only=False)
        for name in node_names:
            node = self.nodes[name]
            for ntr in node.get_connections():
                yield ntr

    def translate(self, translator):
        root_params = self.get_root_parameters()
        new_root_params = [param.rename(translator(param.name)) for param in root_params]
        new_tree = TransformationTree(new_root_params)
        new_tree.reconnect(self, translator=translator)
        return new_tree

    def get_subtree(self, parameters):
        # If the user was not messing with undocumented fields, parameters with the same names
        # should have the same annotations.
        # But if they do not, we better catch it here.
        for param in parameters:
            if param.name in self.root_annotations:
                assert self.root_annotations[param.name] == param.annotation

        new_tree = TransformationTree(parameters)
        new_tree.reconnect(self)
        return new_tree

    def get_kernel_definition(self, kernel_name):
        leaf_params = self.get_leaf_parameters()

        kernel_definition = TEMPLATE.get_def('kernel_definition').render(
            kernel_name, leaf_params, leaf_name=leaf_name)
        leaf_names = [param.name for param in leaf_params]

        return kernel_definition, leaf_names

    def _get_transformation_module(self, annotation, ntr):

        # HACK: Technically, ``module`` attribute is not documented.
        # The reason it is used here is that I need to keep generation of C names for
        # index variable in one place, and the template is the best choice
        # (because they are mostly used there).
        param = Parameter(ntr.connector_node_name, annotation)
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
                tr_args.append(self._get_argobject(connection_name, tr_param.annotation))

        subtree_params = self.get_leaf_parameters([ntr.connector_node_name])

        return Module(
            TEMPLATE.get_def(transformation_def),
            render_kwds=dict(
                tr_snippet=ntr.tr.snippet,
                tr_args=tr_args,
                param=param,
                subtree_params=subtree_params,
                leaf_name=leaf_name))

    def _get_argobject(self, name, annotation, base=False):
        # Takes a base argument name and returns the corresponding Argument object
        # which can be passed to the main kernel.
        # If the name is not in base, it is treated as a leaf.

        node = self.nodes[name]
        param = Parameter(name, annotation)

        if not annotation.array:
            return KernelParameter(param.name, param.annotation.type)

        load_idx = None
        store_idx = None
        load_same = None
        store_same = None
        load_combined_idx = None
        store_combined_idx = None

        if annotation.input:
            if node.input_ntr is None:
                load_idx = Module(
                    TEMPLATE.get_def('leaf_input_macro'),
                    render_kwds=dict(param=param, leaf_name=leaf_name))
            else:
                load_idx = self._get_transformation_module(annotation, node.input_ntr)

            subtree_params = self.get_leaf_parameters([name])

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

        if annotation.output:
            if node.output_ntr is None:
                store_idx = Module(
                    TEMPLATE.get_def('leaf_output_macro'),
                    render_kwds=dict(param=param, leaf_name=leaf_name))
            else:
                store_idx = self._get_transformation_module(annotation, node.output_ntr)

            subtree_params = self.get_leaf_parameters([name])

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
        return [
            self._get_argobject(name, self.root_annotations[name], base=True)
            for name in self.root_names]


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
        attrs = dict(
            load_idx='li', store_idx='si', load_same='ls', store_same='ss',
            load_combined_idx='lci', store_combined_idx='sci')

        attr_str = ", ".join([abbr for name, abbr in attrs.items() if hasattr(self, name)])
        if len(attr_str) > 0:
            attr_str = ", " + attr_str

        return "KernelParameter("+ self.name + attr_str + ")"

    def __str__(self):
        return self._leaf_name
