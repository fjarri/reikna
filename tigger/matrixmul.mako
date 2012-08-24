<%def name="matrixmul(out, a, b)">

${kernel_definition}
{
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

    int matrix_num = by / ${blocks_per_matrix};
    by -= ${blocks_per_matrix} * matrix_num;

    int A_shift = 0;
    int B_shift = 0;
    int C_shift = 0;

    %if batched_a:
        A_shift += matrix_num * ${basis.a_height} * ${basis.a_width};
    %endif
    %if batched_b:
        B_shift += matrix_num * ${basis.a_width} * ${basis.b_width};
    %endif
    C_shift += matrix_num * ${basis.a_height} * ${basis.b_width};

    // Index of the first sub-matrix of A processed by the block
    int aBegin = ${basis.a_width * block_width} * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + ${basis.a_width} - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = ${block_width};

    // Index of the first sub-matrix of B processed by the block
    int bBegin = ${block_width} * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = ${block_width} * ${basis.b_width};

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    ${out.ctype} Csub = ${dtypes.zero_ctr(out.dtype)};

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a_idx = aBegin, b_idx = bBegin, step = 0; a_idx <= aEnd;
        a_idx += aStep, b_idx += bStep, step++)
    {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        int a_x = step * ${block_width} + tx;
        int a_y = by * ${block_width} + ty;
        int b_x = bx * ${block_width} + tx;
        int b_y = step * ${block_width} + ty;

        As[ty * ${block_width} + tx] = (a_x < ${basis.a_width} && a_y < ${basis.a_height})
            ? ${a.load}(a_idx + A_shift + ${basis.a_width} * ty + tx) : ${dtypes.zero_ctr(a.dtype)};
        Bs[ty * ${block_width} + tx] = (b_x < ${basis.b_width} && b_y < ${basis.a_width})
            ? ${b.load}(b_idx + B_shift + ${basis.b_width} * ty + tx) : ${dtypes.zero_ctr(b.dtype)};

        LOCAL_BARRIER;

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < ${block_width}; k++)
            Csub = Csub + ${func.mul(a.dtype, b.dtype, out=out.dtype)}(
                As[ty * ${block_width} + k], Bs[k * ${block_width} + tx]);

        LOCAL_BARRIER;
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c_x = ${block_width} * bx + tx;
    int c_y = ${block_width} * by + ty;
    if(c_y < ${basis.a_height} && c_x < ${basis.b_width})
        ${out.store}(C_shift + ${basis.b_width} * c_y + c_x, Csub);
}

</%def>
