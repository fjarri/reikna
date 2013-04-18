<%!
    INDEX_NAME = "idx"
    VALUE_NAME = "val"
%>

<%def name="argument_list(nodes)">
%for i, node in enumerate(nodes):
<%
    ctype = dtypes.ctype(node.value.dtype)
    name = node.leaf_name
    comma = "" if i == len(nodes) - 1 else ","
%>
    %if node.value.is_array:
        GLOBAL_MEM ${ctype} *${name}${comma}
    %else:
        ${ctype} ${name}${comma}
    %endif
%endfor
</%def>


<%def name="kernel_definition(kernel_name, nodes)">
KERNEL void ${kernel_name}(${argument_list(nodes)})
</%def>


<%def name="leaf_macro(prefix)">
// leaf ${node.type} "${node.name}"
%if node.type == node.INPUT:
#define ${prefix}(${INDEX_NAME}) (${node.leaf_name}[${INDEX_NAME}])
%else:
%if base:
#define ${prefix}(${INDEX_NAME}, ${VALUE_NAME}) ${node.leaf_name}[${INDEX_NAME}] = (${VALUE_NAME})
%else:
#define ${prefix}(${VALUE_NAME}) ${node.leaf_name}[${INDEX_NAME}] = (${VALUE_NAME})
%endif
%endif
</%def>


<%def name="transformation_node(prefix)">
// ${node.type} "${node.name}"
<%
    outtype = dtypes.ctype(node.value.dtype) if node_type == node.OUTPUT else 'void'
    arglist = ", ".join([leaf_node.leaf_name for leaf_node in leaf_nodes])
%>

INLINE WITHIN_KERNEL ${outtype} ${prefix}func(${argument_list(leaf_nodes)}, int ${INDEX_NAME})
{
    ${tr_snippet(*tr_args)}
}

%if node.type == node.INPUT:
#define ${prefix}idx(${INDEX_NAME}) ${prefix}func(${arglist}, ${INDEX_NAME})
#define ${prefix} ${prefix}(${INDEX_NAME})
%else:
%if base:
#define ${prefix}(${INDEX_NAME}, ${VALUE_NAME}) ${prefix}func(${arglist}, ${INDEX_NAME}, ${VALUE_NAME})
%else:
#define ${prefix}(${VALUE_NAME}) ${prefix}func(${arglist}, ${INDEX_NAME}, ${VALUE_NAME})
%endif
%endif
</%def>
