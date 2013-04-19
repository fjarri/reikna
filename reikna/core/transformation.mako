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
// leaf ${node_type} "${node.name}"
%if node_type == node.INPUT:
%if base:
#define ${prefix}(${INDEX_NAME}) (${node.leaf_name}[${INDEX_NAME}])
%else:
#define ${prefix} (${node.leaf_name}[${INDEX_NAME}])
%endif
%else:
%if base:
#define ${prefix}(${INDEX_NAME}, ${VALUE_NAME}) ${node.leaf_name}[${INDEX_NAME}] = (${VALUE_NAME})
%else:
#define ${prefix}(${VALUE_NAME}) ${node.leaf_name}[${INDEX_NAME}] = (${VALUE_NAME})
%endif
%endif
</%def>


<%def name="connector(node)">
%if node.type == node.INPUT:
return
%else:
${VALUE_NAME}
%endif
</%def>


<%def name="transformation_node(prefix)">
// ${node.type} "${node.name}"
<%
    connector_ctype = dtypes.ctype(node.value.dtype)
    outtype = connector_ctype if node.type == node.INPUT else 'void'
    inarg = "" if node.type == node.INPUT else (", " + connector_ctype + " " + VALUE_NAME)
    arglist = ", ".join([leaf_node.leaf_name for leaf_node in leaf_nodes])
%>

INLINE WITHIN_KERNEL ${outtype} ${prefix}func(${argument_list(leaf_nodes)},
    int ${INDEX_NAME} ${inarg})
{
    ${tr_snippet(*tr_args)}
}

%if node.type == node.INPUT:
%if base:
#define ${prefix}(${INDEX_NAME}) ${prefix}func(${arglist}, ${INDEX_NAME})
%else:
#define ${prefix} ${prefix}func(${arglist}, ${INDEX_NAME})
%endif
%else:
%if base:
#define ${prefix}(${INDEX_NAME}, ${VALUE_NAME}) ${prefix}func(${arglist}, ${INDEX_NAME}, ${VALUE_NAME})
%else:
#define ${prefix}(${VALUE_NAME}) ${prefix}func(${arglist}, ${INDEX_NAME}, ${VALUE_NAME})
%endif
%endif
</%def>
