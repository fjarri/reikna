<%def name="reduce(output, input)">

<%
    log2_warp_size = log2(warp_size)
    log2_block_size = log2(block_size)
    if block_size > warp_size:
        smem_size = block_size
    else:
        smem_size = block_size + block_size // 2

    ctype = output.ctype
%>

INLINE WITHIN_KERNEL ${ctype} reduction_op(${ctype} input1, ${ctype} input2)
{
    ${predicate.operation('input1', 'input2')}
}

${kernel_definition}
{
    VIRTUAL_SKIP_THREADS;

    LOCAL_MEM ${ctype} local_mem[${smem_size}];

    int tid = virtual_local_id(0);
    int bid = virtual_group_id(0);
    int part_num = virtual_global_id(1);

    int index_in_part = ${block_size} * bid + tid;

    if(bid == ${blocks_per_part} - 1 && tid >= ${last_block_size})
        local_mem[tid] = ${predicate.empty};
    else
        local_mem[tid] = ${input.load_combined_idx(input_slices)}(part_num, index_in_part);

    LOCAL_BARRIER;

    // 'if(tid)'s will split execution only near the border of warps,
    // so they are not affecting performance (i.e, for each warp there
    // will be only one path of execution anyway)
    %for reduction_pow in range(log2_block_size - 1, log2_warp_size, -1):
        if(tid < ${2 ** reduction_pow})
        {
            local_mem[tid] = reduction_op(local_mem[tid],
                local_mem[tid + ${2 ** reduction_pow}]);
        }
        LOCAL_BARRIER;
    %endfor

    // The following code will be executed inside a single warp, so no
    // shared memory synchronization is necessary
    %if log2_block_size > 0:
    if (tid < ${warp_size}) {
    #ifdef CUDA
    // Fix for Fermi videocards, see Compatibility Guide 1.2.2
    volatile ${ctype} *smem = local_mem;
    #else
    LOCAL_MEM volatile ${ctype} *smem = local_mem;
    #endif

    ${ctype} ttt;
    %for reduction_pow in range(min(log2_warp_size, log2_block_size - 1), -1, -1):
        ttt = reduction_op(smem[tid], smem[tid + ${2 ** reduction_pow}]);
        smem[tid] = ttt;
    %endfor
    }
    %endif

    if (tid == 0)
        ${output.store_combined_idx(output_slices)}(part_num, bid, local_mem[0]);
}

</%def>
