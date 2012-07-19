KERNEL void matrixmul(${signature})
{
    // Storage for sub-matrices of A and B
    // Not using dynamic local memory, because we need (in general) two different types,
    // and creating two views for dynamic char* array does not work.
    // Can cause problems if atype/btype have constructors (they will be called in each thread),
    // but as long as we are using POD types, we will be fine.
    LOCAL_MEM ${ctype.a} As[${block_size ** 2}];
    LOCAL_MEM ${ctype.b} Bs[${block_size ** 2}];

    int bx = GID_0;
    int by = GID_1;
    int tx = LID_0;
    int ty = LID_1;

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
    int aBegin = ${basis.a_width * block_size} * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + ${basis.a_width} - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = ${block_size};

    // Index of the first sub-matrix of B processed by the block
    int bBegin = ${block_size} * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = ${block_size} * ${basis.b_width};

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    ${ctype.out} Csub = ${dtypes.zero_ctr(dtype.out)};

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a_idx = aBegin, b_idx = bBegin, step = 0; a_idx <= aEnd;
        a_idx += aStep, b_idx += bStep, step++)
    {
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        int a_x = step * ${block_size} + tx;
        int a_y = by * ${block_size} + ty;
        int b_x = bx * ${block_size} + tx;
        int b_y = step * ${block_size} + ty;

        As[ty * ${block_size} + tx] = (a_x < ${basis.a_width} && a_y < ${basis.a_height})
            ? ${load.a}(a_idx + A_shift + ${basis.a_width} * ty + tx) : ${dtypes.zero_ctr(dtype.a)};
        Bs[ty * ${block_size} + tx] = (b_x < ${basis.b_width} && b_y < ${basis.a_width})
            ? ${load.b}(b_idx + B_shift + ${basis.b_width} * ty + tx) : ${dtypes.zero_ctr(dtype.b)};

        local_barrier();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < ${block_size}; k++)
            Csub = Csub + ${func.mul(dtype.a, dtype.b, out=dtype.out)}(
                As[ty * ${block_size} + k], Bs[k * ${block_size} + tx]);

        local_barrier();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c_x = ${block_size} * bx + tx;
    int c_y = ${block_size} * by + ty;
    if(c_y < ${basis.a_height} && c_x < ${basis.b_width})
        ${store.out}(C_shift + ${basis.b_width} * c_y + c_x, Csub);
}
