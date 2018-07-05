import weakref

from reikna.helpers import template_def
from reikna.cluda import Snippet
from reikna.core.signature import Signature, Type, Parameter, Annotation
from reikna.core.transformation_modules import leaf_name, node_connector, module_transformation, \
    module_leaf_macro, module_same_indices, module_combined, kernel_declaration, index_cnames


class TransformationParameter(Type):
    """
    Bases: :py:class:`~reikna.core.Type`

    Represents a typed transformation parameter.
    Can be used as a substitute of an array for functions
    which are only interested in array metadata.
    """

    def __init__(self, trf, name, type_):
        Type.__init__(
            self, type_.dtype, shape=type_.shape, strides=type_.strides, offset=type_.offset)
        self._trf = weakref.ref(trf)
        self._name = name

    def belongs_to(self, trf):
        return self._trf() is trf

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
        get_names = lambda ntr: ([], []) if ntr is None else ntr.get_child_names()
        output_before, output_after = get_names(self.output_ntr)
        input_before, input_after = get_names(self.input_ntr)

        if self.output_ntr is None and self.input_ntr is None:
            return [], []
        elif self.output_ntr is None:
            return input_before, input_after
        elif self.input_ntr is None:
            return output_before, output_after
        else:
            return output_before + output_after, input_before + input_after

    def get_connections(self):
        get_conn = lambda ntr: [] if ntr is None else [ntr]
        return get_conn(self.output_ntr) + get_conn(self.input_ntr)


class NodeTransformation:
    """
    Represents a transformation between two nodes
    (an edge of the transformation tree).
    """

    def __init__(self, connector_node_name, trf, node_from_tr):
        self.trf = trf
        self.connector_node_name = connector_node_name
        self.node_from_tr = dict(node_from_tr)

        # Check that all transformation parameter names are represented,
        # and each of them is only mentioned once.
        tr_names_given = set(self.node_from_tr)
        tr_names_available = set(trf.signature.parameters)
        if tr_names_given != tr_names_available:
            raise ValueError(
                "Supplied transformation names {given} "
                " do not fully coincide with the existing ones: {available}".format(
                    given=tr_names_given, available=tr_names_available))

        # If node names were not guaranteed to be different
        # (which is enforced by Computation.connect()),
        # we would have to make the same check for them as well.

        tr_connectors = [
            tr_name for tr_name, node_name in node_from_tr.items()
            if node_name == connector_node_name]
        # There should be only one transformation parameter corresponding to the connector node
        # Not sure under which circumstances it is not true...
        assert len(tr_connectors) == 1
        tr_connector = tr_connectors[0]

        # Taking into account that 'io' parameters are not allowed for transformations.
        self.output = trf.signature.parameters[tr_connector].annotation.input

    def get_child_names(self):
        # Walking the tree conserving the order of parameters in the transformation.
        node_names = [
            self.node_from_tr[tr_param.name]
            for tr_param in self.trf.signature.parameters.values()]
        connector_idx = node_names.index(self.connector_node_name)
        return node_names[:connector_idx], node_names[connector_idx+1:]

    def translate_node_names(self, translator):
        connector_node_name = translator(self.connector_node_name)
        node_from_tr = dict(
            (tr_name, translator(node_name))
            for tr_name, node_name in self.node_from_tr.items())
        return NodeTransformation(connector_node_name, self.trf, node_from_tr)


class TransformationTree:
    """
    A directed acyclic graph of transformations of the kernel arguments.
    """

    def __init__(self, root_parameters):
        # Preserve order of initial root parameters.
        # These can repeat.
        self.root_names = []

        # Keeping whole parameters, because we want to preserve the default values (if any).
        self.root_parameters = {}

        self.nodes = {} # all nodes of the tree
        self.leaf_parameters = {} # nodes available for connection

        for param in root_parameters:
            self.root_names.append(param.name)
            if param.name in self.root_parameters and param != self.root_parameters[param.name]:
                # Could be an 'io' parameter used for separate 'i' and 'o' parameters
                # in a nested computation.
                # Need to check types and merge.

                new_ann = param.annotation
                old_param = self.root_parameters[param.name]
                old_ann = old_param.annotation

                # FIXME: Not sure when these can be raised
                assert old_ann.type == new_ann.type
                assert old_param.default == param.default

                # Given the old_param != param, the only possible combinations of roles are
                # 'i' and 'o', 'i' and 'io', 'o' and 'io'.
                # In all cases the resulting role is 'io'.
                new_param = Parameter(
                    param.name, Annotation(new_ann.type, 'io'), default=param.default)
                self.root_parameters[param.name] = new_param
                self.leaf_parameters[param.name] = new_param
            else:
                self.nodes[param.name] = Node()
                self.root_parameters[param.name] = param
                self.leaf_parameters[param.name] = param

    def _get_subtree_names(self, names, ignore, visited, leaves_only=False):
        """Helper method for traversing the tree."""

        result = []

        for i, name in enumerate(names):
            if name in ignore or name in visited:
                continue

            visited.add(name)
            ignore_in_children = names[i+1:]

            node = self.nodes[name]
            children_before, children_after = node.get_child_names()

            # If the node has a connection yet is still in the leaf parameters
            # (which means it is an i/o node),
            # we want to preserve its position in the list of parameters
            # of the connected transformation.
            # (It helps to preserve the order of the transformation signature
            # when PureParallel computation is created out of it, among other things)
            # This means that we have two subtrees: for the parameters before the connection,
            # and for the parameters after it.

            subtree_before = self._get_subtree_names(
                children_before, ignore_in_children, visited, leaves_only=leaves_only)
            subtree_after = self._get_subtree_names(
                children_after, ignore_in_children, visited, leaves_only=leaves_only)

            if not leaves_only:
                result.append(name)
            result += subtree_before
            if leaves_only and name in self.leaf_parameters:
                result.append(name)
            result += subtree_after

        return result

    def get_subtree_names(self, root_names=None, leaves_only=False):
        if root_names is None:
            root_names = self.root_names
        return self._get_subtree_names(root_names, [], set(), leaves_only=leaves_only)

    def get_root_annotations(self):
        return dict((name, param.annotation) for name, param in self.root_parameters.items())

    def get_root_parameters(self):
        return [self.root_parameters[name] for name in self.root_names]

    def get_leaf_parameters(self, root_names=None):
        leaf_names = self.get_subtree_names(root_names=root_names, leaves_only=True)
        return [self.leaf_parameters[name] for name in leaf_names]

    def _connect(self, ntr):

        # At this point we assume that ``ntr`` describes a valid connection.
        # All sanity checks are performed in ``connect()``.

        for tr_param in ntr.trf.signature.parameters.values():
            node_name = ntr.node_from_tr[tr_param.name]

            if node_name == ntr.connector_node_name:
                ann = self.leaf_parameters[node_name].annotation
                if ann.input and ann.output:
                # splitting the 'io' leaf
                    updated_role = 'i' if ntr.output else 'o'

                    # Since it is an array parameter, we do not need to worry
                    # about preserving the default value (it can't have one).
                    self.leaf_parameters[node_name] = Parameter(
                        node_name, Annotation(ann.type, role=updated_role))
                else:
                # 'i' or 'o' leaf is hidden by the transformation
                    del self.leaf_parameters[node_name]

            else:
                if (node_name in self.leaf_parameters and
                        self.leaf_parameters[node_name].annotation.array):
                    ann = self.leaf_parameters[node_name].annotation
                    if (ann.input and ntr.output) or (ann.output and not ntr.output):
                    # Joining 'i' and 'o' paths into an 'io' leaf.
                    # Since it is an array parameter, we do not need to worry
                    # about preserving the default value (it can't have one).
                        self.leaf_parameters[node_name] = Parameter(
                            node_name, Annotation(ann.type, role='io'))
                else:
                    self.leaf_parameters[node_name] = tr_param.rename(node_name)

            if node_name not in self.nodes:
                self.nodes[node_name] = Node()

        self.nodes[ntr.connector_node_name] = self.nodes[ntr.connector_node_name].connect(ntr)

    def connect(self, comp_connector, trf, comp_from_tr):

        ntr = NodeTransformation(comp_connector, trf, comp_from_tr)

        # Check that the types of connections are correct
        for tr_name, node_name in comp_from_tr.items():
            if node_name not in self.leaf_parameters:
                if node_name == comp_connector:
                    raise ValueError(
                        "Parameter '" + node_name + "' is not a part of the signature")
                elif node_name in self.nodes:
                    raise ValueError(
                        "Parameter '" + node_name + "' is hidden by transformations")

            if node_name not in self.leaf_parameters:
                # If node names could repeat, we would have to check that
                # transformation parameters with incompatible types are not pointing
                # at the same new node.
                continue

            node_ann = self.leaf_parameters[node_name].annotation
            tr_ann = trf.signature.parameters[tr_name].annotation

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

            if ntr.connector_node_name not in self.leaf_parameters:
                continue

            # In the nested tree this particular node may only use one data path
            # (input or output), despite it being 'io' in the parent tree.
            # Thus we only need to reconnect the transformation if such data path exists.
            ann = self.leaf_parameters[ntr.connector_node_name].annotation
            if (ntr.output and ann.output) or (not ntr.output and ann.input):
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
        # Unless the user was not messing with undocumented fields,
        # same names will correspond to the same parameters.
        # But if they do not, we better catch it here.
        subtree_params = []
        for param in parameters:
            if param.name in self.root_parameters:
                assert self.root_parameters[param.name].annotation == param.annotation
                # Not using the parameter that came from the user,
                # because we want to preserve the default value that is saved in our parameter.
                subtree_params.append(self.root_parameters[param.name])
            else:
                subtree_params.append(param)

        new_tree = TransformationTree(subtree_params)
        new_tree.reconnect(self)
        return new_tree

    def get_kernel_declaration(self, kernel_name, skip_constants=False):
        leaf_params = self.get_leaf_parameters()

        if skip_constants:
            leaf_params = [param for param in leaf_params if not param.annotation.constant]

        decl = kernel_declaration(kernel_name, leaf_params)
        leaf_names = [param.name for param in leaf_params]

        return decl, leaf_names

    def _get_transformation_module(self, annotation, ntr):

        param = Parameter(ntr.connector_node_name, annotation)

        tr_args = [Indices(param.annotation.type.shape)]
        connection_names = []
        for tr_param in ntr.trf.signature.parameters.values():
            connection_name = ntr.node_from_tr[tr_param.name]
            connection_names.append(connection_name)

            if connection_name == ntr.connector_node_name:
                if ntr.output:
                    load_same = node_connector(ntr.output)
                    tr_args.append(KernelParameter(
                        param.name, param.annotation.type, load_same=load_same))
                else:
                    store_same = node_connector(ntr.output)
                    tr_args.append(KernelParameter(
                        param.name, param.annotation.type, store_same=store_same))
            else:
                tr_args.append(self._get_kernel_argobject(connection_name, tr_param.annotation))

        subtree_params = self.get_leaf_parameters([ntr.connector_node_name])

        return module_transformation(ntr.output, param, subtree_params, ntr.trf.snippet, tr_args)

    def _get_connection_modules(self, output, name, annotation):

        node = self.nodes[name]
        param = Parameter(name, annotation)
        ntr = node.output_ntr if output else node.input_ntr

        m_idx = None
        m_same = None
        m_combined = None

        if ntr is None:
            m_idx = module_leaf_macro(output, param)
        else:
            m_idx = self._get_transformation_module(annotation, ntr)

        subtree_params = self.get_leaf_parameters([name])

        # FIXME: this module won't work at the base level (that is, not in a trnsformation)
        # unless 'idx' variables were defined.
        # This behavior was enabled for PureParallel.from_trf(), which defines these variables.
        m_same = module_same_indices(output, param, subtree_params, m_idx)

        m_combined = module_combined(output, param, subtree_params, m_idx)

        return m_idx, m_same, m_combined

    def _get_kernel_argobject(self, name, annotation):
        # Returns a parameter object, which can be passed to the main kernel.

        if not annotation.array:
            return KernelParameter(name, annotation.type)

        load_idx, load_same, load_combined_idx = self._get_connection_modules(
            False, name, annotation)
        store_idx, store_same, store_combined_idx = self._get_connection_modules(
            True, name, annotation)

        return KernelParameter(
            name, annotation.type,
            load_idx=load_idx,
            store_idx=store_idx,
            load_same=load_same,
            store_same=store_same,
            load_combined_idx=load_combined_idx,
            store_combined_idx=store_combined_idx)

    def get_kernel_argobjects(self):
        return [
            self._get_kernel_argobject(name, self.root_parameters[name].annotation)
            for name in self.root_names]


class Indices:
    """
    Encapsulates the information about index variables available for the snippet.
    """

    def __init__(self, shape):
        """__init__()""" # hide the signature from Sphinx
        self._names = index_cnames(shape)

    def __getitem__(self, dim):
        """
        Returns the name of the index varibale for the dimension ``dim``.
        """
        return self._names[dim]

    def all(self):
        """
        Returns the comma-separated list of all index variable names
        (useful for passing the guiding indices verbatim in a load or store call).
        """
        return ', '.join(self._names)


class KernelParameter:
    """
    Providing an interface for accessing kernel arguments in a template.
    Depending on the parameter type, and whether it is used
    inside a computation or a transformation template,
    can have different load/store attributes available.

    .. py:attribute:: name

        Parameter name

    .. py:attribute:: shape
    .. py:attribute:: dtype
    .. py:attribute:: ctype
    .. py:attribute:: strides
    .. py:attribute:: offset

        Same as in :py:class:`~reikna.core.Type`.

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

        self._type = type_

        self.shape = type_.shape
        self.strides = type_.strides
        self.offset = type_.offset
        self.dtype = type_.dtype
        self.ctype = type_.ctype

        self._leaf_name = leaf_name(name)
        self.name = name

        if load_idx is not None:
            self.load_idx = load_idx
        if store_idx is not None:
            self.store_idx = store_idx
        if load_same is not None:
            self.load_same = load_same
        if store_same is not None:
            self.store_same = store_same
        if load_combined_idx is not None:
            self.load_combined_idx = load_combined_idx
        if store_combined_idx is not None:
            self.store_combined_idx = store_combined_idx

    def __process_modules__(self, process):
        kwds = {}
        attrs = (
            'load_idx', 'store_idx', 'load_same', 'store_same',
            'load_combined_idx', 'store_combined_idx')
        for attr in attrs:
            if hasattr(self, attr):
                kwds[attr] = process(getattr(self, attr))

        return KernelParameter(self.name, process(self._type), **kwds)

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
