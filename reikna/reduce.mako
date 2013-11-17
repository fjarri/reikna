<%def name="reduce(kernel_declaration, output, input)">

<%
    log2_block_size = log2(block_size)
    smem_size = block_size

    ctype = output.ctype

    fields = dtypes.flatten_dtype(output.dtype)
    paths = [('.' if len(path) > 0 else '') + '.'.join(path) for path, _ in fields]
    ctypes = [dtypes.ctype(dtype) for _, dtype in fields]
    suffixes = ['_' + '_'.join(path) for path, _ in fields]
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

    VSIZE_T tid = virtual_local_id(1);
    VSIZE_T bid = virtual_group_id(1);
    VSIZE_T part_num = virtual_global_id(0);

    VSIZE_T index_in_part = ${block_size} * bid + tid;
    const ${ctype} empty = ${dtypes.c_constant(empty)};

    ${ctype} v;
    if(bid == ${blocks_per_part} - 1 && tid >= ${last_block_size})
        v = empty;
    else
        v = ${input.load_combined_idx(input_slices)}(part_num, index_in_part);
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
