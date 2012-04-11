${define_mul_func('mul_func', bp.a_dtype, bp.b_dtype, bp.out_dtype)}

<%
    outtype_name = helpers.ctype(bp.out_dtype)
    atype_name = helpers.ctype(bp.a_dtype)
    btype_name = helpers.ctype(bp.b_dtype)
%>

KERNEL void matrixmul(GLOBAL_MEM ${outtype_name}* C, GLOBAL_MEM ${atype_name}* A, GLOBAL_MEM ${btype_name}* B)
{
    // Storage for sub-matrices of A and B
    // Not using dynamic local memory, because we need (in general) two different types,
    // and creating two views for dynamic char* array does not work.
    // Can cause problems if atype/btype have constructors (they will be called in each thread),
    // but as long as we are using POD types, we will be fine.
    LOCAL_MEM ${atype_name} As[${dp.block_size ** 2}];
    LOCAL_MEM ${btype_name} Bs[${dp.block_size ** 2}];

    int bx = GID_0;
    int by = GID_1;
    int tx = LID_0;
    int ty = LID_1;

    int matrix_num = by / ${dp.blocks_per_matrix};
    by -= ${dp.blocks_per_matrix} * matrix_num;

    %if batched_a:
        A += matrix_num * ${bp.a_height} * ${bp.a_width};
    %endif
    %if batched_b:
        B += matrix_num * ${bp.a_width} * ${bp.b_width};
    %endif
    C += matrix_num * ${bp.a_height} * ${bp.b_width};

    // Index of the first sub-matrix of A processed by the block
    int aBegin = ${bp.a_width * dp.block_size} * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + ${bp.a_width} - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = ${dp.block_size};

    // Index of the first sub-matrix of B processed by the block
    int bBegin = ${dp.block_size} * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = ${dp.block_size} * ${bp.b_width};

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    ${outtype_name} Csub = ${helpers.zero_ctr(bp.out_dtype)};

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin, step = 0; a <= aEnd; a += aStep, b += bStep, step++)
    {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        int a_x = step * ${dp.block_size} + tx;
        int a_y = by * ${dp.block_size} + ty;
        int b_x = bx * ${dp.block_size} + tx;
        int b_y = step * ${dp.block_size} + ty;

        As[ty * ${dp.block_size} + tx] = (a_x < ${bp.a_width} && a_y < ${bp.a_height})
            ? A[a + ${bp.a_width} * ty + tx] : ${helpers.zero_ctr(bp.a_dtype)};
        Bs[ty * ${dp.block_size} + tx] = (b_x < ${bp.b_width} && b_y < ${bp.a_width})
            ? B[b + ${bp.b_width} * ty + tx] : ${helpers.zero_ctr(bp.b_dtype)};

        local_barrier();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < ${dp.block_size}; k++)
            Csub = Csub + mul_func(As[ty * ${dp.block_size} + k], Bs[k * ${dp.block_size} + tx]);

        local_barrier();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c_x = ${dp.block_size} * bx + tx;
    int c_y = ${dp.block_size} * by + ty;
    if(c_y < ${bp.a_height} && c_x < ${bp.b_width})
        C[${bp.b_width} * c_y + c_x] = Csub;
}
