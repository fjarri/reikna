<%def name="reduce(output, input)">

${code_functions(output, input)}

<%
    log2_warp_size = log2(warp_size)
    log2_block_size = log2(block_size)
    if block_size > warp_size:
        smem_size = block_size
    else:
        smem_size = block_size + block_size / 2

    ctype = output.ctype
%>

INLINE WITHIN_KERNEL ${ctype} _reduction_op(${ctype} input1, ${ctype} input2)
{
    ${code_kernel(output, input)}
}

${kernel_definition}
{
    LOCAL_MEM ${ctype} shared_mem[${smem_size}];

    int tid = get_local_id(0);
    int bid = get_group_id(0);

    int part_length = ${(blocks_per_part - 1) * block_size + last_block_size};
    int part_num = bid / ${blocks_per_part};
    int index_in_part = ${block_size} * (bid % ${blocks_per_part}) + tid;

    if(bid % ${blocks_per_part} == ${blocks_per_part} - 1 && tid >= ${last_block_size})
        shared_mem[tid] = ${dtypes.zero_ctr(basis.dtype)};
    else
        shared_mem[tid] = ${input.load}(part_length * part_num + index_in_part);

    LOCAL_BARRIER;

    // 'if(tid)'s will split execution only near the border of warps,
    // so they are not affecting performance (i.e, for each warp there
    // will be only one path of execution anyway)
    %for reduction_pow in xrange(log2_block_size - 1, log2_warp_size, -1):
        if(tid < ${2 ** reduction_pow})
        {
            shared_mem[tid] = _reduction_op(shared_mem[tid],
                shared_mem[tid + ${2 ** reduction_pow}]);
        }
        LOCAL_BARRIER;
    %endfor

    // The following code will be executed inside a single warp, so no
    // shared memory synchronization is necessary
    %if log2_block_size > 0:
    if (tid < ${warp_size}) {
    #ifdef CUDA
    // Fix for Fermi videocards, see Compatibility Guide 1.2.2
    volatile ${ctype} *smem = shared_mem;
    #else
    LOCAL_MEM ${ctype} *smem = shared_mem;
    #endif

    ${ctype} ttt;
    %for reduction_pow in xrange(min(log2_warp_size, log2_block_size - 1), -1, -1):
        ttt = _reduction_op(smem[tid], smem[tid + ${2 ** reduction_pow}]);
        smem[tid] = ttt;
    %endfor
    }
    %endif

    if (tid == 0)
        ${output.store}(bid, shared_mem[0]);
}

</%def>
