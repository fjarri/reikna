<%def name="scan(kernel_declaration, output, input, wg_totals)">

<%
    ctype = output.ctype
%>

#define CONFLICT_FREE_OFFSET(n) ((n) >> ${log_num_banks})

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    LOCAL_MEM ${ctype} temp[${wg_size + (wg_size >> log_num_banks)}];

    VSIZE_T thid = virtual_local_id(1);
    VSIZE_T batch_id = virtual_global_id(0);
    VSIZE_T scan_id = virtual_global_id(1);
    VSIZE_T wg_id = virtual_group_id(1);
    VSIZE_T global_offset = scan_id * ${seq_size};

    // Sequential scan
    %for i in range(seq_size):
    ${ctype} seq_data${i};
    %endfor

    %for i in range(seq_size):
    if (scan_id * ${seq_size} + ${i} < ${scan_size})
        seq_data${i} = ${input.load_combined_idx(slices)}(
            batch_id, global_offset + ${i});
    else
        seq_data${i} = 0;
    %endfor

    ${ctype} seq_total;
    {
        ${ctype} t;
        seq_total = seq_data0;
        seq_data0 = 0;
        %for i in range(1, seq_size):
        t = seq_data${i};
        seq_data${i} = seq_total;
        seq_total = seq_total + t;
        %endfor
    }

    // load input into shared memory
    temp[thid + CONFLICT_FREE_OFFSET(thid)] = seq_total;

    // build sum in place up the tree
    %for d in range(log_wg_size - 1, -1, -1):
    <%
        offset = 2**(log_wg_size - d - 1)
    %>
    {
        LOCAL_BARRIER;

        if (thid < ${2**d})
        {
            int ai = ${offset}*(2*thid+1)-1;
            int bi = ${offset}*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
    }
    %endfor

    // save and clear the last element (the sum of all the elements in the block)
    ${ctype} wg_total;
    if (thid == 0) {
        wg_total = temp[${wg_size} - 1 + CONFLICT_FREE_OFFSET(${wg_size} - 1)];
        ${wg_totals.store_idx}(batch_id, wg_id, wg_total);
        temp[${wg_size} - 1 + CONFLICT_FREE_OFFSET(${wg_size} - 1)] = 0;
    }

    %for d in range(log_wg_size):
    <%
        offset = 2**(log_wg_size - d - 1)
    %>
    {
        LOCAL_BARRIER;
        if (thid < ${2**d})
        {
            int ai = ${offset}*(2*thid+1)-1;
            int bi = ${offset}*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            ${ctype} t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    %endfor
    LOCAL_BARRIER;

    ${ctype} res = temp[thid + CONFLICT_FREE_OFFSET(thid)];
    %if not exclusive:

        if (thid == 0)
        {
            int this_part_size = wg_id == (${wg_totals_size} - 1) ?
                ${last_part_size} : ${wg_size * seq_size};
            ${output.store_combined_idx(slices)}(
                batch_id, global_offset + this_part_size - 1, wg_total);
        }
        else
        {
            if (global_offset - 1 < ${scan_size})
            {
                ${output.store_combined_idx(slices)}(
                    batch_id, global_offset - 1, res + seq_data0);
            }
        }
        %for i in range(1, seq_size):
        if (global_offset + ${i} - 1 < ${scan_size})
        {
            ${output.store_combined_idx(slices)}(
                batch_id, global_offset + ${i} - 1, res + seq_data${i});
        }
        %endfor

    %else:

        %for i in range(seq_size):
        if (global_offset + ${i} < ${scan_size})
        {
            ${output.store_combined_idx(slices)}(
                batch_id, global_offset + ${i}, res + seq_data${i});
        }
        %endfor

    %endif
}

</%def>


<%def name="add_wg_totals(kernel_declaration, output, per_wg_results, wg_totals)">

<%
    ctype = output.ctype
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    VSIZE_T batch_id = virtual_global_id(0);
    VSIZE_T scan_id = virtual_global_id(1);

    ${ctype} per_wg_result = ${per_wg_results.load_combined_idx(slices)}(batch_id, scan_id);
    ${ctype} wg_total = ${wg_totals.load_idx}(batch_id, scan_id / ${wg_size * seq_size});
    ${output.store_combined_idx(slices)}(batch_id, scan_id, per_wg_result + wg_total);
}

</%def>
