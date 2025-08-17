from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from grunnur import DefTemplate, Module, Snippet, dtypes

from .. import helpers

if TYPE_CHECKING:
    from .signature import Parameter
    from .transformation import Transformation


VALUE_NAME = "_val"
INDEX_NAME = "_idx"


def leaf_name(name: str) -> str:
    return "_leaf_" + name


def index_cnames(shape: Sequence[int]) -> list[str]:
    return [INDEX_NAME + str(i) for i in range(len(shape))]


def index_cnames_seq(param: "Parameter", *, qualified: bool = False) -> list[str]:
    names = index_cnames(param.annotation.type.shape)
    if qualified:
        return ["VSIZE_T " + name for name in names]
    return names


def flat_index_expr(param: "Parameter") -> str:
    type_ = param.annotation.type

    # TODO: Can we support it?
    if type_.first_element_offset % type_.dtype.itemsize != 0:
        raise ValueError("Offset must be a multiple of the itemsize")
    item_offset = type_.first_element_offset // type_.dtype.itemsize

    if len(type_.shape) == 0:
        return str(item_offset)

    # TODO: Assuming that all strides are multiples of dtype.itemsize.
    # This can change with custom strides, and we will have
    # to cast device pointer to bytes and back.
    # Need to investigate what happens in this case on some concrete example.
    if not all(stride % type_.dtype.itemsize == 0 for stride in type_.strides):
        raise ValueError(
            "Some of the strides "
            + str(type_.strides)
            + "are not multiples of the itemsize"
            + str(type_.dtype.itemsize)
        )

    item_strides = [stride // type_.dtype.itemsize for stride in type_.strides]

    names = index_cnames(param.annotation.type.shape)

    return (
        " + ".join(
            [
                "(" + name + ")" + " * " + "(" + str(stride) + ")"
                for name, stride in zip(names, item_strides, strict=False)
            ]
        )
        + " + ("
        + str(item_offset)
        + ")"
    )


def param_cname(param: "Parameter", *, qualified: bool = False) -> str:
    # Note that if ``param`` has a struct type,
    # its .annotation.ctype attribute can be a module.
    # In that case ``str()`` has to be called explicitly for ``ctype``
    # to get the module prefix.
    name = "_leaf_" + param.name
    if qualified:
        ctype = param.annotation.ctype
        if param.annotation.is_array:
            qualifier = "CONSTANT_MEM" if param.annotation.constant else "GLOBAL_MEM"
            return qualifier + " " + str(ctype) + " *" + name

        return str(ctype) + " " + name

    return name


def param_cnames_seq(parameters: "Iterable[Parameter]", *, qualified: bool = False) -> list[str]:
    return [param_cname(p, qualified=qualified) for p in parameters]


_snippet_kernel_declaration = DefTemplate.from_string(
    "kernel_declaration",
    [],
    "KERNEL void ${kernel_name}(${', '.join(param_cnames_seq(parameters, qualified=True))})",
)


def kernel_declaration(kernel_name: str, parameters: "Iterable[Parameter]") -> Snippet:
    return Snippet(
        _snippet_kernel_declaration,
        render_globals=dict(
            param_cnames_seq=param_cnames_seq, kernel_name=kernel_name, parameters=parameters
        ),
    )


def node_connector(*, output: bool) -> str:
    if output:
        return VALUE_NAME
    return VALUE_NAME + " ="


_module_transformation = DefTemplate.from_string(
    "transformation_function",
    ["prefix"],
    """
    // ${'output' if output else 'input'} transformation node for "${name}"
    <%
        value_param = [str(connector_ctype) + ' ' + VALUE_NAME] if output else []
        value = [VALUE_NAME] if output else []

        signature = (
            param_cnames_seq(subtree_parameters, qualified=True) +
            q_indices +
            value_param)
    %>
    INLINE FUNCTION ${'void' if output else connector_ctype} ${prefix}func(
        ${",\\n".join(signature)})
    {
        %if not output:
        ${connector_ctype} ${VALUE_NAME};
        %endif

        ${tr_snippet(*tr_args)}

        %if not output:
        return ${VALUE_NAME};
        %endif
    }
    <%
    %>
    #define ${prefix}(${', '.join(nq_indices + value)}) ${prefix}func(\\
        ${', '.join(nq_params + nq_indices + value)})
    """,
)


def module_transformation(
    output: "Transformation",
    param: "Parameter",
    subtree_parameters: "list[Parameter]",
    tr_snippet: Snippet,
    tr_args: list[Any],
) -> Module:
    return Module(
        _module_transformation,
        render_globals=dict(
            output=output,
            name=param.name,
            param_cnames_seq=param_cnames_seq,
            subtree_parameters=subtree_parameters,
            q_indices=index_cnames_seq(param, qualified=True),
            VALUE_NAME=VALUE_NAME,
            nq_params=param_cnames_seq(subtree_parameters),
            nq_indices=index_cnames_seq(param),
            connector_ctype=param.annotation.ctype,
            tr_snippet=tr_snippet,
            tr_args=tr_args,
        ),
    )


_module_leaf_macro = DefTemplate.from_string(
    "leaf_macro",
    ["prefix"],
    """
    // leaf ${'output' if output else 'input'} macro for "${name}"
    %if output:
    #define ${prefix}(${', '.join(index_seq + [VALUE_NAME])}) \
        ${lname}[${index_expr}] = (${VALUE_NAME})
    %else:
    #define ${prefix}(${', '.join(index_seq)}) (${lname}[${index_expr}])
    %endif
    """,
)


def module_leaf_macro(param: "Parameter", *, output: bool) -> Module:
    return Module(
        _module_leaf_macro,
        render_globals=dict(
            output=output,
            name=param.name,
            VALUE_NAME=VALUE_NAME,
            lname=leaf_name(param.name),
            index_seq=index_cnames_seq(param),
            index_expr=flat_index_expr(param),
        ),
    )


_module_same_indices = DefTemplate.from_string(
    "same_indices_macro",
    ["prefix"],
    """
    // ${'output' if output else 'input'} for a transformation for "${name}"
    %if output:
    #define ${prefix}(${VALUE_NAME}) ${module_idx}(${', '.join(nq_indices + [VALUE_NAME])})
    %else:
    #define ${prefix} ${module_idx}(${', '.join(nq_indices)})
    %endif
    """,
)


def module_same_indices(
    param: "Parameter",
    subtree_parameters: "Iterable[Parameter]",
    module_idx: Module,
    *,
    output: bool,
) -> Module:
    return Module(
        _module_same_indices,
        render_globals=dict(
            output=output,
            name=param.name,
            VALUE_NAME=VALUE_NAME,
            module_idx=module_idx,
            nq_indices=index_cnames_seq(param),
            nq_params=param_cnames_seq(subtree_parameters),
        ),
    )


_snippet_disassemble_combined = Snippet.from_string(
    ["shape", "slices", "indices", "combined_indices"],
    """
    %for combined_index, slice_len in enumerate(slices):
    <%
        index_start = sum(slices[:combined_index])
        index_end = index_start + slice_len
    %>
        %for index in range(index_start, index_end):
        <%
            stride = product(shape[index+1:index_end])
        %>
        VSIZE_T ${indices[index]} = ${combined_indices[combined_index]} / ${stride};
        %if index != index_end - 1:
        ${combined_indices[combined_index]} -= ${indices[index]} * ${stride};
        %endif
        %endfor
    %endfor
    """,
    render_globals=dict(product=helpers.product),
)

_module_combined = DefTemplate.from_string(
    "combined_transformation",
    ["prefix", "slices"],
    """
    <%
        value_param = [str(connector_ctype) + ' ' + VALUE_NAME] if output else []
        value = [VALUE_NAME] if output else []

        combined_indices = ['_c_idx' + str(i) for i in range(len(slices))]
        q_combined_indices = ['VSIZE_T ' + ind for ind in combined_indices]
        nq_combined_indices = combined_indices
        signature = (
            param_cnames_str(subtree_parameters, qualified=True) +
            q_combined_indices +
            value_param)
    %>
    INLINE FUNCTION ${'void' if output else connector_ctype} ${prefix}func(
        ${', '.join(signature)})
    {
        ${disassemble(shape, slices, indices, combined_indices)}

        %if not output:
        return
        %endif
        ${module_idx}(${', '.join(nq_indices + value)});
    }
    <%
        value = [VALUE_NAME] if output else []
    %>
    #define ${prefix}(${', '.join(nq_combined_indices + value)}) ${prefix}func(\\
        ${', '.join(nq_params + nq_combined_indices + value)})
    """,
)


def module_combined(
    param: "Parameter",
    subtree_parameters: "Iterable[Parameter]",
    module_idx: Module,
    *,
    output: bool,
) -> Module:
    return Module(
        _module_combined,
        render_globals=dict(
            output=output,
            VALUE_NAME=VALUE_NAME,
            shape=param.annotation.type.shape,
            module_idx=module_idx,
            disassemble=_snippet_disassemble_combined,
            connector_ctype=dtypes.ctype(param.annotation.type.dtype),
            nq_indices=index_cnames_seq(param),
            q_indices=index_cnames_seq(param, qualified=True),
            param_cnames_str=param_cnames_seq,
            subtree_parameters=subtree_parameters,
            nq_params=param_cnames_seq(subtree_parameters),
            indices=index_cnames(param.annotation.type.shape),
        ),
    )
