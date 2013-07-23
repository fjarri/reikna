from reikna.helpers import *
from reikna.cluda import Module, Snippet


VALUE_NAME = "_val"
INDEX_NAME = "_idx"


def leaf_name(name):
    return "_leaf_" + name


def index_cnames(param):
    type = param.annotation.type
    return [INDEX_NAME + str(i) for i in range(len(type.shape))]


def index_cnames_str(param, qualified=False):
    names = index_cnames(param)
    if qualified:
        names = ["int " + name for name in names]
    return ", ".join(names)


def flat_index_expr(param):
    # FIXME: Assuming that all strides are multiples of dtype.itemsize,
    # which is true as long as we use primitive types from numpy.
    # This can change with custom structures, and we will have
    # to cast device pointer to bytes and back.
    # Need to investigate what happens in this case on some concrete example.

    type = param.annotation.type
    item_strides = [stride // type.dtype.itemsize for stride in type.strides]

    names = index_cnames(param)

    return " + ".join([
        name + " * " + str(stride)
        for name, stride in zip(names, item_strides)])


def param_cname(p, qualified=False):
    name = "_leaf_" + p.name
    if qualified:
        ctype = p.annotation.type.ctype
        if p.annotation.array:
            return "GLOBAL_MEM " + ctype + " *" + name
        else:
            return ctype + " " + name
    else:
        return name


def param_cnames_str(params, qualified=False):
    return ", ".join([param_cname(p, qualified=qualified) for p in params])


def kernel_definition(kernel_name, params):
    return "KERNEL void {kernel_name}({cnames_str})".format(
        kernel_name=kernel_name, cnames_str=param_cnames_str(params, qualified=True))


def node_input_connector():
    return VALUE_NAME + " ="


def node_output_connector():
    return VALUE_NAME


def node_input_transformation(param, subtree_params, tr_snippet, tr_args):
    return Module.create(
        """
        // input transformation node for "${name}"
        INLINE WITHIN_KERNEL ${connector_ctype} ${prefix}func(
            ${q_params},
            ${q_indices})
        {
            ${connector_ctype} ${VALUE_NAME};

            ${tr_snippet(*tr_args)}

            return ${VALUE_NAME};
        }
        #define ${prefix}(${nq_indices}) ${prefix}func(${nq_params}, ${nq_indices})
        """,
        render_kwds=dict(
            name=param.name,
            q_params=param_cnames_str(subtree_params, qualified=True),
            q_indices=index_cnames_str(param, qualified=True),
            VALUE_NAME=VALUE_NAME,
            nq_params=param_cnames_str(subtree_params),
            nq_indices=index_cnames_str(param),
            connector_ctype=param.annotation.type.ctype,
            tr_snippet=tr_snippet,
            tr_args=tr_args))


def node_output_transformation(param, subtree_params, tr_snippet, tr_args):
    return Module.create(
        """
        // output transformation node for "${name}"
        INLINE WITHIN_KERNEL void ${prefix}func(
            ${q_params},
            ${q_indices},
            ${connector_ctype} ${VALUE_NAME})
        {
            ${tr_snippet(*tr_args)}
        }
        #define ${prefix}(${nq_indices}, ${VALUE_NAME}) """ +
            """${prefix}func(${nq_params}, ${nq_indices}, ${VALUE_NAME})""",
        render_kwds=dict(
            name=param.name,
            q_params=param_cnames_str(subtree_params, qualified=True),
            q_indices=index_cnames_str(param, qualified=True),
            VALUE_NAME=VALUE_NAME,
            nq_params=param_cnames_str(subtree_params),
            nq_indices=index_cnames_str(param),
            connector_ctype=param.annotation.type.ctype,
            tr_snippet=tr_snippet,
            tr_args=tr_args))


def leaf_input_macro(param):
    return Module.create(
        """
        // leaf input macro for "${name}"
        #define ${prefix}(${index_str}) (${lname}[${index_expr}])
        """,
        render_kwds=dict(
            name=param.name,
            lname=leaf_name(param.name),
            index_str=index_cnames_str(param),
            index_expr=flat_index_expr(param)))


def leaf_output_macro(param):
    return Module.create(
        """
        // leaf output macro for "${name}"
        #define ${prefix}(${index_str}, ${VALUE_NAME}) ${lname}[${index_expr}] = (${VALUE_NAME})
        """,
        render_kwds=dict(
            name=param.name,
            lname=leaf_name(param.name),
            index_str=index_cnames_str(param),
            index_expr=flat_index_expr(param),
            VALUE_NAME=VALUE_NAME))


def node_input_same_indices(param, subtree_params, load_idx):
    return Module.create("""
        // input for a transformation for "${name}"
        #define ${prefix} ${load_idx}(${nq_indices})
        """,
        render_kwds=dict(
            name=param.name,
            load_idx=load_idx,
            nq_indices=index_cnames_str(param),
            nq_params=param_cnames_str(subtree_params)))


def node_output_same_indices(param, subtree_params, store_idx):
    return Module.create("""
        // output from a transformation for "${name}"
        #define ${prefix}(${VALUE_NAME}) ${store_idx}(${nq_indices}, ${VALUE_NAME})
        """,
        render_kwds=dict(
            name=param.name,
            store_idx=store_idx,
            VALUE_NAME=VALUE_NAME,
            nq_indices=index_cnames_str(param),
            nq_params=param_cnames_str(subtree_params)))


disassemble_combined = Snippet.create(
    lambda shape, slices, indices, combined_indices: """
    %for combined_index, slice_len in enumerate(slices):
    <%
        index_start = sum(slices[:combined_index])
        index_end = index_start + slice_len
    %>
        %for index in range(index_start, index_end):
        <%
            stride = product(shape[index+1:index_end])
        %>
        int ${indices[index]} = ${combined_indices[combined_index]} / ${stride};
        ${combined_indices[combined_index]} -= ${indices[index]} * ${stride};
        %endfor
    %endfor
    """,
    render_kwds=dict(product=product))


def node_input_combined(param, subtree_params, load_idx):
    return Module.create(
        lambda prefix, slices: """
        <%
            combined_indices=['_c_idx' + str(i) for i in range(len(slices))]
            q_combined_indices=", ".join(['int ' + ind for ind in combined_indices])
            nq_combined_indices=", ".join(combined_indices)
        %>
        INLINE WITHIN_KERNEL ${connector_ctype} ${prefix}func(
            ${q_params},
            ${q_combined_indices})
        {
            ${disassemble(shape, slices, indices, combined_indices)}
            return ${load_idx}(${nq_indices});
        }
        #define ${prefix}(${nq_combined_indices}) ${prefix}func(${nq_params}, ${nq_combined_indices})
        """,
        render_kwds=dict(
            shape=param.annotation.type.shape,
            load_idx=load_idx,
            disassemble=disassemble_combined,
            connector_ctype=param.annotation.type.ctype,
            nq_indices=index_cnames_str(param),
            q_indices=index_cnames_str(param, qualified=True),
            q_params=param_cnames_str(subtree_params, qualified=True),
            nq_params=param_cnames_str(subtree_params),
            indices=index_cnames(param)))


def node_output_combined(param, subtree_params, store_idx):
    return Module.create(
        lambda prefix, slices: ("""
        <%
            combined_indices=['_c_idx' + str(i) for i in range(len(slices))]
            q_combined_indices=", ".join(['int ' + ind for ind in combined_indices])
            nq_combined_indices=", ".join(combined_indices)
        %>
        INLINE WITHIN_KERNEL void ${prefix}func(
            ${q_params},
            ${q_combined_indices},
            ${connector_ctype} ${VALUE_NAME})
        {
            ${disassemble(shape, slices, indices, combined_indices)}
            ${store_idx}(${nq_indices}, ${VALUE_NAME});
        }
        #define ${prefix}(${nq_combined_indices}, ${VALUE_NAME}) """ +
            """${prefix}func(${nq_params}, ${nq_combined_indices}, ${VALUE_NAME})"""),
        render_kwds=dict(
            shape=param.annotation.type.shape,
            store_idx=store_idx,
            VALUE_NAME=VALUE_NAME,
            disassemble=disassemble_combined,
            connector_ctype=param.annotation.type.ctype,
            nq_indices=index_cnames_str(param),
            q_indices=index_cnames_str(param, qualified=True),
            q_params=param_cnames_str(subtree_params, qualified=True),
            nq_params=param_cnames_str(subtree_params),
            indices=index_cnames(param)))
