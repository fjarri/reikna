<%def name="shift1Dinplace(kernel_declaration, output, input)">
<%
    ctype = dtypes.ctype(output.dtype)
%>
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    VSIZE_T index = virtual_global_id(0);
    if (index < ${NX // 2})
    {
        // Save the first value
        ${ctype} regTemp = ${input.load_idx}(index);

        // Swap the first element
        ${output.store_idx}(index, ${input.load_idx}(index + ${NX // 2}));

        // Swap the second one
        ${output.store_idx}(index + ${NX // 2}, regTemp);
    }
}

</%def>
