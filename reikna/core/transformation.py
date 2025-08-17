import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy
from grunnur import ArrayMetadata, AsArrayMetadata, DefTemplate, Snippet

from ..core.signature import Annotation, Parameter, Signature
from ..core.transformation_modules import (
    index_cnames,
    kernel_declaration,
    leaf_name,
    module_combined,
    module_leaf_macro,
    module_same_indices,
    module_transformation,
    node_connector,
)

if TYPE_CHECKING:
    from grunnur import Module


class TransformationParameter:
    """
    Represents a typed transformation parameter.
    Can be used as a substitute of an array for functions
    which are only interested in array metadata.
    """

    def __init__(self, trf: "Transformation", name: str, type: ArrayMetadata | numpy.dtype[Any]):
        self._trf = weakref.ref(trf)
        self._name = name
        self._type = type

    def belongs_to(self, trf: "Transformation") -> bool:
        return self._trf() is trf

    def __str__(self) -> str:
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

    def __init__(
        self,
        parameters: Sequence[Parameter],
        code: str,
        render_kwds: Mapping[str, Any] | None = None,
        connectors: Sequence[str] | None = None,
    ):
        for param in parameters:
            if param.annotation.input and param.annotation.output:
                raise ValueError(
                    "Transformation cannot have 'io' parameters ('" + param.name + "')"
                )

        self.signature = Signature(parameters)

        # TODO: remove when we switch to using `.parameter` everywhere
        for param in self.signature._reikna_parameters.values():  # noqa: SLF001
            setattr(
                self, param.name, TransformationParameter(self, param.name, param.annotation.type)
            )

        self.parameter = TransformationParameterContainer(self, parameters)

        if connectors is not None:
            self.connectors = connectors
        else:
            self.connectors = [param.name for param in parameters if param.annotation.is_array]

        tr_param_names = ["idxs"] + [param.name for param in self.signature.parameters.values()]
        self.snippet = Snippet(
            DefTemplate.from_string("transformation", tr_param_names, code),
            render_globals=render_kwds or {},
        )


class TransformationParameterContainer:
    """A convenience object with ``TransformationParameter`` attributes."""

    def __init__(self, parent: "Transformation", parameters: Iterable[Parameter]):
        self._param_objs = {
            param.name: TransformationParameter(parent, param.name, param.annotation.type)
            for param in parameters
        }

    def __getattr__(self, name: str) -> TransformationParameter:
        return self._param_objs[name]


class Node:
    """Represents a node in the transformation tree."""

    def __init__(
        self,
        input_ntr: "NodeTransformation | None" = None,
        output_ntr: "NodeTransformation | None" = None,
    ):
        self.input_ntr = input_ntr
        self.output_ntr = output_ntr

    def connect(self, ntr: "NodeTransformation") -> "Node":
        if ntr.output:
            return Node(input_ntr=self.input_ntr, output_ntr=ntr)
        return Node(input_ntr=ntr, output_ntr=self.output_ntr)

    def get_child_names(self) -> tuple[list[str], list[str]]:
        def get_names(ntr: "NodeTransformation | None") -> tuple[list[str], list[str]]:
            return ([], []) if ntr is None else ntr.get_child_names()

        output_before, output_after = get_names(self.output_ntr)
        input_before, input_after = get_names(self.input_ntr)

        if self.output_ntr is None and self.input_ntr is None:
            return [], []
        if self.output_ntr is None:
            return input_before, input_after
        if self.input_ntr is None:
            return output_before, output_after
        return output_before + output_after, input_before + input_after

    def get_connections(self) -> "list[NodeTransformation]":
        def get_conn(ntr: "NodeTransformation | None") -> "list[NodeTransformation]":
            return [] if ntr is None else [ntr]

        return get_conn(self.output_ntr) + get_conn(self.input_ntr)


class NodeTransformation:
    """
    Represents a transformation between two nodes
    (an edge of the transformation tree).
    """

    def __init__(
        self, connector_node_name: str, trf: Transformation, node_from_tr: Mapping[str, str]
    ):
        self.trf = trf
        self.connector_node_name = connector_node_name
        self.node_from_tr = dict(node_from_tr)

        # Check that all transformation parameter names are represented,
        # and each of them is only mentioned once.
        tr_names_given = set(self.node_from_tr)
        tr_names_available = set(trf.signature.parameters)
        if tr_names_given != tr_names_available:
            raise ValueError(
                f"Supplied transformation names {tr_names_given} "
                f" do not fully coincide with the existing ones: {tr_names_available}"
            )

        # If node names were not guaranteed to be different
        # (which is enforced by Computation.connect()),
        # we would have to make the same check for them as well.

        tr_connectors = [
            tr_name
            for tr_name, node_name in node_from_tr.items()
            if node_name == connector_node_name
        ]

        # TODO: Not sure under which circumstances it is not true...
        if len(tr_connectors) != 1:
            raise ValueError(
                "There should be only one transformation parameter "
                "corresponding to the connector node"
            )
        tr_connector = tr_connectors[0]

        # Taking into account that 'io' parameters are not allowed for transformations.
        self.output = trf.signature.parameters[tr_connector].annotation.input

    def get_child_names(self) -> tuple[list[str], list[str]]:
        # Walking the tree conserving the order of parameters in the transformation.
        node_names = [
            self.node_from_tr[tr_param.name] for tr_param in self.trf.signature.parameters.values()
        ]
        connector_idx = node_names.index(self.connector_node_name)
        return node_names[:connector_idx], node_names[connector_idx + 1 :]

    def translate_node_names(self, translator: Callable[[str], str]) -> "NodeTransformation":
        connector_node_name = translator(self.connector_node_name)
        node_from_tr = {
            tr_name: translator(node_name) for tr_name, node_name in self.node_from_tr.items()
        }
        return NodeTransformation(connector_node_name, self.trf, node_from_tr)


class TransformationTree:
    """A directed acyclic graph of transformations of the kernel arguments."""

    def __init__(self, root_parameters: Iterable[Parameter]):
        # Preserve order of initial root parameters.
        # These can repeat.
        # TODO: dictionaries preserve order in modern Python
        self.root_names = []

        # Keeping whole parameters, because we want to preserve the default values (if any).
        self.root_parameters: dict[str, Parameter] = {}

        self.nodes = {}  # all nodes of the tree
        self.leaf_parameters = {}  # nodes available for connection

        for param in root_parameters:
            self.root_names.append(param.name)
            if param.name in self.root_parameters and param != self.root_parameters[param.name]:
                # Could be an 'io' parameter used for separate 'i' and 'o' parameters
                # in a nested computation.
                # Need to check types and merge.

                new_ann = param.annotation
                old_param = self.root_parameters[param.name]
                old_ann = old_param.annotation

                # TODO: Not sure when these can be raised
                if old_ann.type != new_ann.type:
                    raise ValueError(f"Type mismatch for two parameters with name `{param.name}`")
                if old_param.default != param.default:
                    raise ValueError(
                        f"Default value mismatch for two parameters with name `{param.name}`"
                    )

                # Given the old_param != param, the only possible combinations of roles are
                # 'i' and 'o', 'i' and 'io', 'o' and 'io'.
                # In all cases the resulting role is 'io'.
                new_param = Parameter(
                    param.name, Annotation(new_ann.type, "io"), default=param.default
                )
                self.root_parameters[param.name] = new_param
                self.leaf_parameters[param.name] = new_param
            else:
                self.nodes[param.name] = Node()
                self.root_parameters[param.name] = param
                self.leaf_parameters[param.name] = param

    def _get_subtree_names(
        self,
        names: Sequence[str],
        ignore: set[str],
        visited: set[str],
        *,
        leaves_only: bool = False,
    ) -> list[str]:
        """Helper method for traversing the tree."""
        result = []

        for i, name in enumerate(names):
            if name in ignore or name in visited:
                continue

            visited.add(name)
            ignore_in_children = set(names[i + 1 :])

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
                children_before, ignore_in_children, visited, leaves_only=leaves_only
            )
            subtree_after = self._get_subtree_names(
                children_after, ignore_in_children, visited, leaves_only=leaves_only
            )

            if not leaves_only:
                result.append(name)
            result += subtree_before
            if leaves_only and name in self.leaf_parameters:
                result.append(name)
            result += subtree_after

        return result

    def get_subtree_names(
        self, root_names: Sequence[str] | None = None, *, leaves_only: bool = False
    ) -> list[str]:
        if root_names is None:
            root_names = self.root_names
        return self._get_subtree_names(root_names, set(), set(), leaves_only=leaves_only)

    def get_root_annotations(self) -> dict[str, Annotation]:
        return {name: param.annotation for name, param in self.root_parameters.items()}

    def get_root_parameters(self) -> list[Parameter]:
        return [self.root_parameters[name] for name in self.root_names]

    def get_leaf_parameters(self, root_names: Sequence[str] | None = None) -> list[Parameter]:
        leaf_names = self.get_subtree_names(root_names=root_names, leaves_only=True)
        return [self.leaf_parameters[name] for name in leaf_names]

    def _connect(self, ntr: NodeTransformation) -> None:
        # At this point we assume that ``ntr`` describes a valid connection.
        # All sanity checks are performed in ``connect()``.

        for tr_param in ntr.trf.signature._reikna_parameters.values():  # noqa: SLF001
            node_name = ntr.node_from_tr[tr_param.name]

            if node_name == ntr.connector_node_name:
                ann = self.leaf_parameters[node_name].annotation
                if ann.input and ann.output:
                    # splitting the 'io' leaf
                    updated_role = "i" if ntr.output else "o"

                    # Since it is an array parameter, we do not need to worry
                    # about preserving the default value (it can't have one).
                    self.leaf_parameters[node_name] = Parameter(
                        node_name, Annotation(ann.type, role=updated_role)
                    )
                else:
                    # 'i' or 'o' leaf is hidden by the transformation
                    del self.leaf_parameters[node_name]

            elif (
                node_name in self.leaf_parameters
                and self.leaf_parameters[node_name].annotation.is_array
            ):
                ann = self.leaf_parameters[node_name].annotation
                if (ann.input and ntr.output) or (ann.output and not ntr.output):
                    # Joining 'i' and 'o' paths into an 'io' leaf.
                    # Since it is an array parameter, we do not need to worry
                    # about preserving the default value (it can't have one).
                    self.leaf_parameters[node_name] = Parameter(
                        node_name, Annotation(ann.type, role="io")
                    )
            else:
                self.leaf_parameters[node_name] = tr_param.rename(node_name)

            if node_name not in self.nodes:
                self.nodes[node_name] = Node()

        self.nodes[ntr.connector_node_name] = self.nodes[ntr.connector_node_name].connect(ntr)

    def connect(  # noqa: C901
        self, comp_connector: str, trf: Transformation, comp_from_tr: Mapping[str, str]
    ) -> None:
        ntr = NodeTransformation(comp_connector, trf, comp_from_tr)

        # Check that the types of connections are correct
        for tr_name, node_name in comp_from_tr.items():
            if node_name not in self.leaf_parameters:
                if node_name == comp_connector:
                    raise ValueError("Parameter '" + node_name + "' is not a part of the signature")
                if node_name in self.nodes:
                    raise ValueError("Parameter '" + node_name + "' is hidden by transformations")

            if node_name not in self.leaf_parameters:
                # If node names could repeat, we would have to check that
                # transformation parameters with incompatible types are not pointing
                # at the same new node.
                continue

            node_ann = self.leaf_parameters[node_name].annotation
            tr_ann = trf.signature.parameters[tr_name].annotation

            if tr_ann.type != node_ann.type:
                raise ValueError(
                    "Incompatible types of the transformation parameter "
                    f"'{tr_name}' ({tr_ann.type}) and the node '{node_name}' ({node_ann.type})"
                )

            # No more to check in the case of scalars
            if not tr_ann.is_array:
                continue

            if node_name == comp_connector:
                if ntr.output and not node_ann.output:
                    raise ValueError("'" + node_name + "' is not an output node")
                if not ntr.output and not node_ann.input:
                    raise ValueError("'" + node_name + "' is not an input node")
            elif ntr.output and node_ann.output:
                raise ValueError(
                    f"Cannot connect transformation parameter '{tr_name}' "
                    f"to an existing output node '{node_name}'"
                )

        self._connect(ntr)

    def reconnect(
        self, other_tree: "TransformationTree", translator: Callable[[str], str] | None = None
    ) -> None:
        for ntr in other_tree.connections():
            ntr_translated = ntr.translate_node_names(translator) if translator is not None else ntr

            if ntr_translated.connector_node_name not in self.leaf_parameters:
                continue

            # In the nested tree this particular node may only use one data path
            # (input or output), despite it being 'io' in the parent tree.
            # Thus we only need to reconnect the transformation if such data path exists.
            ann = self.leaf_parameters[ntr_translated.connector_node_name].annotation
            if (ntr_translated.output and ann.output) or (not ntr_translated.output and ann.input):
                self._connect(ntr_translated)

    def connections(self) -> Iterator[NodeTransformation]:
        node_names = self.get_subtree_names(leaves_only=False)
        for name in node_names:
            node = self.nodes[name]
            yield from node.get_connections()

    def translate(self, translator: Callable[[str], str]) -> "TransformationTree":
        root_params = self.get_root_parameters()
        new_root_params = [param.rename(translator(param.name)) for param in root_params]
        new_tree = TransformationTree(new_root_params)
        new_tree.reconnect(self, translator=translator)
        return new_tree

    def get_subtree(self, parameters: Iterable[Parameter]) -> "TransformationTree":
        # Unless the user was not messing with undocumented fields,
        # same names will correspond to the same parameters.
        # But if they do not, we better catch it here.
        subtree_params = []
        for param in parameters:
            if param.name in self.root_parameters:
                if self.root_parameters[param.name].annotation != param.annotation:
                    raise ValueError(f"Annotation mismatch for the parameter `{param.name}`")
                # Not using the parameter that came from the user,
                # because we want to preserve the default value that is saved in our parameter.
                subtree_params.append(self.root_parameters[param.name])
            else:
                subtree_params.append(param)

        new_tree = TransformationTree(subtree_params)
        new_tree.reconnect(self)
        return new_tree

    def get_kernel_declaration(
        self, kernel_name: str, *, skip_constants: bool = False
    ) -> tuple[Snippet, list[str]]:
        leaf_params = self.get_leaf_parameters()

        if skip_constants:
            leaf_params = [param for param in leaf_params if not param.annotation.constant]

        decl = kernel_declaration(kernel_name, leaf_params)
        leaf_names = [param.name for param in leaf_params]

        return decl, leaf_names

    def _get_transformation_module(
        self, annotation: Annotation, ntr: NodeTransformation
    ) -> "Module":
        param = Parameter(ntr.connector_node_name, annotation)

        tr_args: list[Any] = [Indices(param.annotation.type.shape)]
        connection_names = []
        for tr_param in ntr.trf.signature._reikna_parameters.values():  # noqa: SLF001
            connection_name = ntr.node_from_tr[tr_param.name]
            connection_names.append(connection_name)

            if connection_name == ntr.connector_node_name:
                if ntr.output:
                    load_same = node_connector(output=ntr.output)
                    tr_args.append(
                        KernelParameter(param.name, param.annotation.type, load_same=load_same)
                    )
                else:
                    store_same = node_connector(output=ntr.output)
                    tr_args.append(
                        KernelParameter(param.name, param.annotation.type, store_same=store_same)
                    )
            else:
                tr_args.append(self._get_kernel_argobject(connection_name, tr_param.annotation))

        subtree_params = self.get_leaf_parameters([ntr.connector_node_name])

        return module_transformation(ntr.output, param, subtree_params, ntr.trf.snippet, tr_args)

    def _get_connection_modules(
        self,
        name: str,
        annotation: Annotation,
        *,
        output: bool,
    ) -> "tuple[Module, Module, Module]":
        node = self.nodes[name]
        param = Parameter(name, annotation)
        ntr = node.output_ntr if output else node.input_ntr

        m_idx = None
        m_same = None
        m_combined = None

        if ntr is None:
            m_idx = module_leaf_macro(param, output=output)
        else:
            m_idx = self._get_transformation_module(annotation, ntr)

        subtree_params = self.get_leaf_parameters([name])

        # TODO: this module won't work at the base level (that is, not in a transformation)
        # unless 'idx' variables were defined.
        # This behavior was enabled for PureParallel.from_trf(), which defines these variables.
        m_same = module_same_indices(param, subtree_params, m_idx, output=output)

        m_combined = module_combined(param, subtree_params, m_idx, output=output)

        return m_idx, m_same, m_combined

    def _get_kernel_argobject(self, name: str, annotation: Annotation) -> "KernelParameter":
        # Returns a parameter object, which can be passed to the main kernel.

        if not annotation.is_array:
            return KernelParameter(name, annotation)

        load_idx, load_same, load_combined_idx = self._get_connection_modules(
            name,
            annotation,
            output=False,
        )
        store_idx, store_same, store_combined_idx = self._get_connection_modules(
            name,
            annotation,
            output=True,
        )

        return KernelParameter(
            name,
            annotation,
            load_idx=load_idx,
            store_idx=store_idx,
            load_same=load_same,
            store_same=store_same,
            load_combined_idx=load_combined_idx,
            store_combined_idx=store_combined_idx,
        )

    def get_kernel_argobjects(self) -> "list[KernelParameter]":
        return [
            self._get_kernel_argobject(name, self.root_parameters[name].annotation)
            for name in self.root_names
        ]


class Indices:
    """Encapsulates the information about index variables available for the snippet."""

    def __init__(self, shape: Sequence[int]):
        """__init__()"""  # hide the signature from Sphinx
        self._names = index_cnames(shape)

    def __getitem__(self, dim: int) -> str:
        """Returns the name of the index varibale for the dimension ``dim``."""
        return self._names[dim]

    def all(self) -> str:
        """
        Returns the comma-separated list of all index variable names
        (useful for passing the guiding indices verbatim in a load or store call).
        """
        return ", ".join(self._names)


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

    def __init__(
        self,
        name: str,
        annotation: Annotation,
        load_idx: "Module | None" = None,
        store_idx: "Module | None" = None,
        load_same: "Module | str | None" = None,
        store_same: "Module | str | None" = None,
        load_combined_idx: "Module | None" = None,
        store_combined_idx: "Module | None" = None,
    ):
        """__init__()"""  # hide the signature from Sphinx
        self._annotation = annotation

        self.ctype = annotation.ctype

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

    @property
    def _metadata(self) -> ArrayMetadata:
        # TODO: should we have two separate kernel parameter types,
        # one for arrays, and one for scalars?
        if isinstance(self._annotation.type, ArrayMetadata):
            return self._annotation.type
        raise TypeError("Scalar kernel parameter has no array metadata")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._metadata.shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._metadata.strides

    @property
    def offset(self) -> int:
        return self._metadata.first_element_offset

    @property
    def dtype(self) -> numpy.dtype[Any]:
        tp = self._annotation.type
        return tp.dtype if isinstance(tp, ArrayMetadata) else tp

    def __repr__(self) -> str:
        attrs = dict(
            load_idx="li",
            store_idx="si",
            load_same="ls",
            store_same="ss",
            load_combined_idx="lci",
            store_combined_idx="sci",
        )

        attr_str = ", ".join([abbr for name, abbr in attrs.items() if hasattr(self, name)])
        if len(attr_str) > 0:
            attr_str = ", " + attr_str

        return "KernelParameter(" + self.name + attr_str + ")"

    def __str__(self) -> str:
        return self._leaf_name
