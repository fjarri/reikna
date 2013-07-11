<%def name="matrixmul(output, a, b)">

${kernel_definition}
{
    VIRTUAL_SKIP_THREADS;

    // Storage for sub-matrices of A and B
    // Not using dynamic local memory, because we need (in general) two different types,
    // and creating two views for dynamic char* array does not work.
    // Can cause problems if atype/btype have constructors (they will be called in each thread),
    // but as long as we are using POD types, we will be fine.
    LOCAL_MEM ${a.ctype} As[${block_width ** 2}];
    LOCAL_MEM ${b.ctype} Bs[${block_width ** 2}];

    int bx = virtual_group_id(0);
    int by = virtual_group_id(1);
    int tx = virtual_local_id(0);
    int ty = virtual_local_id(1);
    int matrix_num = virtual_global_id(2);

    %if batched_a:
    int A_num = matrix_num;
    %else:
    int A_num = 0;
    %endif

    %if batched_b:
    int B_num = matrix_num;
    %else:
    int B_num = 0;
    %endif

    int C_num = matrix_num;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    ${output.ctype} Csub = ${dtypes.zero_ctr(output.dtype)};

    int c_x = ${block_width} * bx + tx;
    int c_y = ${block_width} * by + ty;
    bool in_c = (c_y < ${a_height} && c_x < ${b_width});

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int step = 0; step < ${num_steps}; step++)
    {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        int a_x = step * ${block_width} + tx;
        int a_y = by * ${block_width} + ty;
        int b_x = bx * ${block_width} + tx;
        int b_y = step * ${block_width} + ty;

        As[ty * ${block_width} + tx] = (a_x < ${a_width} && a_y < ${a_height})
            ? ${a.load_combined_idx(a_slices)}(A_num, a_y, a_x) : ${dtypes.zero_ctr(a.dtype)};
        Bs[ty * ${block_width} + tx] = (b_x < ${b_width} && b_y < ${a_width})
            ? ${b.load_combined_idx(b_slices)}(B_num, b_y, b_x) : ${dtypes.zero_ctr(b.dtype)};

        LOCAL_BARRIER;

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        if (in_c)
        {
            for (int k = 0; k < ${block_width}; k++)
                Csub = Csub + ${mul}(As[ty * ${block_width} + k], Bs[k * ${block_width} + tx]);
        }

        LOCAL_BARRIER;
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    if(in_c)
        ${output.store_combined_idx(output_slices)}(C_num, c_y, c_x, Csub);
}

</%def>
