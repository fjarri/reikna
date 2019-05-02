<%def name="fftshift_inplace(kernel_declaration, output, input)">
<%
    dimensions = len(output.shape)
    idx_names = ['index' + str(idx) for idx in range(dimensions)]
    new_idx_names = ['new_index' + str(idx) for idx in range(dimensions)]
%>
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    %for dim in range(dimensions):
    VSIZE_T ${idx_names[dim]} = virtual_global_id(${dim});
    %endfor

    %for dim in range(dimensions):
    VSIZE_T ${new_idx_names[dim]} =
        ${idx_names[dim]}
        %if dim in axes:
        +
            %if dim != axes[0]:
            (${idx_names[dim]} < ${output.shape[dim] // 2} ?
                ${output.shape[dim] // 2} :
                ${-output.shape[dim] // 2})
            %else:
            ${output.shape[dim] // 2}
            %endif
        %endif
        ;
    %endfor

    ${output.ctype} val1 = ${input.load_idx}(${', '.join(idx_names)});
    ${output.ctype} val2 = ${input.load_idx}(${', '.join(new_idx_names)});

    ${output.store_idx}(${', '.join(idx_names)}, val2);
    ${output.store_idx}(${', '.join(new_idx_names)}, val1);
}
</%def>


<%def name="fftshift_outplace(kernel_declaration, output, input, inverse)">
<%
    dimensions = len(output.shape)
    idx_names = ['index' + str(idx) for idx in range(dimensions)]
    new_idx_names = ['new_index' + str(idx) for idx in range(dimensions)]
%>
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    %for dim in range(dimensions):
    VSIZE_T ${idx_names[dim]} = virtual_global_id(${dim});
    %endfor

    %for dim in range(dimensions):
    VSIZE_T ${new_idx_names[dim]} =
        ${idx_names[dim]}
        %if dim in axes:
            %if output.shape[dim] % 2 == 0:
            + (${idx_names[dim]} < ${output.shape[dim] // 2} ?
                ${output.shape[dim] // 2} :
                ${-output.shape[dim] // 2})
            %else:
            + (${idx_names[dim]} <= ${output.shape[dim] // 2} ?
                ${output.shape[dim] // 2} :
                ${-(output.shape[dim] // 2 + 1)})
            %endif
        %endif
        ;
    %endfor

    if (${inverse})
    {
        VSIZE_T temp;
        %for dim in range(dimensions):
        temp = ${new_idx_names[dim]};
        ${new_idx_names[dim]} = ${idx_names[dim]};
        ${idx_names[dim]} = temp;
        %endfor
    }

    ${output.ctype} val = ${input.load_idx}(${', '.join(idx_names)});
    ${output.store_idx}(${', '.join(new_idx_names)}, val);
}
</%def>
