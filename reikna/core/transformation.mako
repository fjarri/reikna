<%!
    INDEX_NAME = "_idx"
    VALUE_NAME = "_val"

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
%>


<%def name="kernel_definition(kernel_name, params)">
KERNEL void ${kernel_name}(${param_cnames_str(params, qualified=True)})
</%def>


<%def name="leaf_input_macro(prefix)">
// leaf input macro for "${param.name}"
#define ${prefix}(${index_cnames_str(param)}) (${leaf_name(param.name)}[${flat_index_expr(param)}])
</%def>


<%def name="leaf_output_macro(prefix)">
// leaf output macro for "${param.name}"
#define ${prefix}(${index_cnames_str(param)}, ${VALUE_NAME}) ${leaf_name(param.name)}[${flat_index_expr(param)}] = (${VALUE_NAME})
</%def>


<%def name="node_input_connector()">
return \
</%def>

<%def name="node_output_connector()">
${VALUE_NAME}\
</%def>


<%def name="node_input_transformation(prefix)">
<%
    connector_ctype = param.annotation.type.ctype
    nq_indices = index_cnames_str(param)
    q_indices = index_cnames_str(param, qualified=True)
    nq_params = param_cnames_str(subtree_params)
    q_params = param_cnames_str(subtree_params, qualified=True)
%>
// input transformation node for "${param.name}"
INLINE WITHIN_KERNEL ${connector_ctype} ${prefix}func(
    ${q_params},
    ${q_indices})
{
    ${tr_snippet(*tr_args)}
}
#define ${prefix}(${nq_indices}) ${prefix}func(${nq_params}, ${nq_indices})
</%def>


<%def name="node_input_same_indices(prefix)">
<%
    nq_indices = index_cnames_str(param)
    nq_params = param_cnames_str(subtree_params)
%>
// input for a transformation for "${param.name}"
#define ${prefix} ${load_idx}(${nq_indices})
</%def>


<%def name="disassemble_combined(combined_indices, indices, slices, shape)">
%for combined_index, slice_len in enumerate(slices):
<%
    index_start = sum(slices[:combined_index])
    index_end = index_start + slice_len
%>
    %for index in range(index_start, index_end):
    <%
        stride = helpers.product(shape[index+1:index_end])
    %>
    int ${indices[index]} = ${combined_indices[combined_index]} / ${stride};
    ${combined_indices[combined_index]} -= ${indices[index]} * ${stride};
    %endfor
%endfor
</%def>


<%def name="node_input_combined(prefix, slices)">
<%
    connector_ctype = param.annotation.type.ctype
    nq_indices = index_cnames_str(param)
    q_indices = index_cnames_str(param, qualified=True)
    q_params = param_cnames_str(subtree_params, qualified=True)
    nq_params = param_cnames_str(subtree_params)

    indices = index_cnames(param)
    combined_indices = ['c_idx' + str(i) for i in range(len(slices))]
    q_combined_indices = ", ".join(['int ' + ind for ind in combined_indices])
    nq_combined_indices = ", ".join(combined_indices)
%>
INLINE WITHIN_KERNEL ${connector_ctype} ${prefix}func(
    ${q_params},
    ${q_combined_indices})
{
    ${disassemble_combined(combined_indices, indices, slices, param.annotation.type.shape)}
    return ${load_idx}(${nq_indices});
}
#define ${prefix}(${nq_combined_indices}) ${prefix}func(${nq_params}, ${nq_combined_indices})
</%def>


<%def name="node_output_combined(prefix, slices)">
<%
    connector_ctype = param.annotation.type.ctype
    nq_indices = index_cnames_str(param)
    q_indices = index_cnames_str(param, qualified=True)
    q_params = param_cnames_str(subtree_params, qualified=True)
    nq_params = param_cnames_str(subtree_params)

    indices = index_cnames(param)
    combined_indices = ['c_idx' + str(i) for i in range(len(slices))]
    q_combined_indices = ", ".join(['int ' + ind for ind in combined_indices])
    nq_combined_indices = ", ".join(combined_indices)
%>
INLINE WITHIN_KERNEL ${connector_ctype} ${prefix}func(
    ${q_params},
    ${q_combined_indices},
    ${connector_ctype} ${VALUE_NAME})
{
    ${disassemble_combined(combined_indices, indices, slices, param.annotation.type.shape)}
    ${store_idx}(${nq_indices}, ${VALUE_NAME});
}
#define ${prefix}(${nq_combined_indices}, ${VALUE_NAME}) ${prefix}func(${nq_params}, ${nq_combined_indices}, ${VALUE_NAME})
</%def>


<%def name="node_output_transformation(prefix)">
<%
    connector_ctype = param.annotation.type.ctype
    nq_indices = index_cnames_str(param)
    q_indices = index_cnames_str(param, qualified=True)
    nq_params = param_cnames_str(subtree_params)
    q_params = param_cnames_str(subtree_params, qualified=True)
%>
// output transformation node for "${param.name}"
INLINE WITHIN_KERNEL void ${prefix}func(
    ${q_params},
    ${q_indices},
    ${connector_ctype} ${VALUE_NAME})
{
    ${tr_snippet(*tr_args)}
}
#define ${prefix}(${nq_indices}, ${VALUE_NAME}) ${prefix}func(${nq_params}, ${nq_indices}, ${VALUE_NAME})
</%def>


<%def name="node_output_same_indices(prefix)">
<%
    nq_indices = index_cnames_str(param)
    nq_params = param_cnames_str(subtree_params)
%>
// output from a transformation for "${param.name}"
#define ${prefix}(${VALUE_NAME}) ${store_idx}(${nq_indices}, ${VALUE_NAME})
</%def>
