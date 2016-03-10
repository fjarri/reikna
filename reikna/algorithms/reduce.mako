<%def name="reduce(kernel_declaration, output, input)">

<%
    log2_block_size = log2(block_size)
    smem_size = block_size

    ctype = output.ctype

    fields = dtypes.flatten_dtype(output.dtype)
    paths = [dtypes.c_path(path) for path, _ in fields]
    ctypes = [dtypes.ctype(dtype) for _, dtype in fields]
    suffixes = ['_' + '_'.join(str(elem) for elem in path) for path, _ in fields]
%>

INLINE WITHIN_KERNEL ${ctype} reduction_op(${ctype} input1, ${ctype} input2)
{
    ${operation('input1', 'input2')}
}

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    %for ct, suffix in zip(ctypes, suffixes):
    LOCAL_MEM ${ct} local_mem${suffix}[${smem_size}];
    %endfor

    const VSIZE_T tid = virtual_local_id(1);
    const VSIZE_T bid = virtual_group_id(1);
    const VSIZE_T part_num = virtual_global_id(0);

    const VSIZE_T index_in_part = ${block_size * seq_size} * bid + tid;
    const ${ctype} empty = ${dtypes.c_constant(empty)};

    ${ctype} v;
    %for i in range(seq_size):
    <%
        conds = []
        if blocks_per_part > 1:
            conds.append("bid <" + str(blocks_per_part - 1))
        if last_block_size > i * block_size:
            conds.append("tid < " + str(last_block_size - i * block_size))
    %>
        %if len(conds) > 0:
        if(${" || ".join(conds)})
        %endif
        {
            const ${ctype} t =
                ${input.load_combined_idx(input_slices)}(
                    part_num, index_in_part + ${i * block_size});
            ## Do not call reduction_op() if it is not necessary.
            ## May matter if it has complicated logic.
            %if i == 0:
            v = t;
            %else:
            v = reduction_op(v, t);
            %endif
        }
        %if i == 0:
        else
        {
            v = empty;
        }
        %endif

    %endfor

    %for path, suffix in zip(paths, suffixes):
    local_mem${suffix}[tid] = v${path};
    %endfor
    LOCAL_BARRIER;

    // We could use the volatile trick here and execute the last several iterations
    // (that fit in a single warp) without LOCAL_BARRIERs, but it gives only
    // a minor performance boost, and works only for some platforms (and only for simple types).
    %for reduction_pow in range(log2_block_size - 1, -1, -1):
        if(tid < ${2 ** reduction_pow})
        {
            ${ctype} val1, val2;
            %for path, suffix in zip(paths, suffixes):
            val1${path} = local_mem${suffix}[tid];
            val2${path} = local_mem${suffix}[tid + ${2 ** reduction_pow}];
            %endfor
            const ${ctype} val = reduction_op(val1, val2);

            %for path, suffix in zip(paths, suffixes):
            local_mem${suffix}[tid] = val${path};
            %endfor
        }
        LOCAL_BARRIER;
    %endfor

    if (tid == 0)
    {
        %for path, suffix in zip(paths, suffixes):
        v${path} = local_mem${suffix}[0];
        %endfor

        ${output.store_combined_idx(output_slices)}(part_num, bid, v);
    }
}

</%def>
