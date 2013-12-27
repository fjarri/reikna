<%def name="matrixmul(kernel_declaration, output, a, b)">

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    // Storage for sub-matrices of A and B
    // Not using dynamic local memory, because we need (in general) two different types,
    // and creating two views for dynamic char* array does not work.
    // Can cause problems if atype/btype have constructors (they will be called in each thread),
    // but as long as we are using POD types, we will be fine.
    LOCAL_MEM ${a.ctype} As[${block_width ** 2} + ${block_width}];
    LOCAL_MEM ${b.ctype} Bs[${block_width ** 2} + ${block_width}];

    const VSIZE_T bx = virtual_group_id(2);
    const VSIZE_T by = virtual_group_id(1);
    const VSIZE_T tx = virtual_local_id(2);
    const VSIZE_T ty = virtual_local_id(1);
    const VSIZE_T matrix_num = virtual_global_id(0);

    %if batched_a:
    const VSIZE_T A_num = matrix_num;
    %else:
    const VSIZE_T A_num = 0;
    %endif

    %if batched_b:
    const VSIZE_T B_num = matrix_num;
    %else:
    const VSIZE_T B_num = 0;
    %endif

    const VSIZE_T C_num = matrix_num;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    ${output.ctype} Csub = ${dtypes.zero_ctr(output.dtype)};

    const VSIZE_T c_x = ${block_width} * bx + tx;
    const VSIZE_T c_y = ${block_width} * by + ty;
    const bool in_c = (c_y < ${output.shape[-2]} && c_x < ${output.shape[-1]});

    <% a_step_dim = 'a_y' if transposed_a else 'a_x' %>
    %if transposed_a:
    const VSIZE_T a_x = by * ${block_width} + tx;
    VSIZE_T a_y = ty;
    %else:
    const VSIZE_T a_y = by * ${block_width} + ty;
    VSIZE_T a_x = tx;
    %endif

    <% b_step_dim = 'b_x' if transposed_b else 'b_y' %>
    %if transposed_b:
    const VSIZE_T b_y = bx * ${block_width} + ty;
    VSIZE_T b_x = tx;
    %else:
    const VSIZE_T b_x = bx * ${block_width} + tx;
    VSIZE_T b_y = ty;
    %endif

    const int store_idx = ty * (${block_width} + 1) + tx;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (VSIZE_T step = 0; step < ${num_steps}; step++)
    {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix

        As[store_idx] = (a_x < ${a.shape[-1]} && a_y < ${a.shape[-2]})
            ? ${a.load_combined_idx(a_slices)}(A_num, a_y, a_x)
            : ${dtypes.zero_ctr(a.dtype)};
        Bs[store_idx] = (b_x < ${b.shape[-1]} && b_y < ${b.shape[-2]})
            ? ${b.load_combined_idx(b_slices)}(B_num, b_y, b_x)
            : ${dtypes.zero_ctr(b.dtype)};

        LOCAL_BARRIER;

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        if (in_c)
        {
            for (unsigned int k = 0; k < ${block_width}; k++)
                Csub = Csub + ${mul}(
                    %if transposed_a:
                    As[k * (${block_width} + 1) + ty],
                    %else:
                    As[ty * (${block_width} + 1) + k],
                    %endif

                    %if transposed_b:
                    Bs[tx * (${block_width} + 1) + k]
                    %else:
                    Bs[k * (${block_width} + 1) + tx]
                    %endif
                    );
        }

        LOCAL_BARRIER;

        ${a_step_dim} += ${block_width};
        ${b_step_dim} += ${block_width};
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    if(in_c)
        ${output.store_combined_idx(output_slices)}(C_num, c_y, c_x, Csub);
}

</%def>
