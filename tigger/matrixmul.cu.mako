%if env.api == 'cuda':
    // taken from pycuda._cluda
    #define local_barrier() __syncthreads()

    #define WITHIN_KERNEL __device__
    #define KERNEL extern "C" __global__
    #define GLOBAL_MEM /* empty */
    #define LOCAL_MEM __shared__
    #define LOCAL_MEM_DYNAMIC extern __shared__
    #define LOCAL_MEM_ARG /* empty */

    #define LID_0 threadIdx.x
    #define LID_1 threadIdx.y
    #define LID_2 threadIdx.z

    #define GID_0 blockIdx.x
    #define GID_1 blockIdx.y
    #define GID_2 blockIdx.z

    #define LSIZE_0 blockDim.x
    #define LSIZE_1 blockDim.y
    #define LSIZE_2 blockDim.z

%elif env.api == 'ocl':
    // taken from pyopencl._cluda
    #define local_barrier() barrier(CLK_LOCAL_MEM_FENCE)

    #define WITHIN_KERNEL /* empty */
    #define KERNEL __kernel
    #define GLOBAL_MEM __global
    #define LOCAL_MEM __local
    #define LOCAL_MEM_DYNAMIC __local
    #define LOCAL_MEM_ARG __local

    #define LID_0 get_local_id(0)
    #define LID_1 get_local_id(1)
    #define LID_2 get_local_id(2)

    #define GID_0 get_group_id(0)
    #define GID_1 get_group_id(1)
    #define GID_2 get_group_id(2)

    // TODO: does this forceful enabling of double precision somehow change
    // the performance for single precision?
    #if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64: enable
    #endif

%endif

%if env.api == 'cuda':
    #define COMPLEX_CTR(T) make_##T
%elif env.api == 'ocl':
    #define COMPLEX_CTR(T) (##T)
%endif

## These operators are supported by OpenCL
%if env.api == 'cuda':
%for tp in ('float2', 'double2'):
    WITHIN_KERNEL ${tp} operator+(${tp} a, ${tp} b) { return COMPLEX_CTR(${tp})(a.x + b.x, a.y + b.y); }
    WITHIN_KERNEL ${tp} operator-(${tp} a, ${tp} b) { return COMPLEX_CTR(${tp})(a.x - b.x, a.y - b.y); }
    WITHIN_KERNEL ${tp} operator+(${tp} a) { return a; }
    WITHIN_KERNEL ${tp} operator-(${tp} a) { return COMPLEX_CTR(${tp})(-a.x, -a.y); }
%endfor
%endif


<%def name="define_mul_func(name, a_dtype, b_dtype, out_dtype)">
WITHIN_KERNEL ${helpers.ctype(out_dtype)} ${name}(${helpers.ctype(a_dtype)} a, ${helpers.ctype(b_dtype)} b)
{
    <%
        if not helpers.is_complex(a_dtype) and not helpers.is_complex(b_dtype):
            result = '(a * b)'
        elif helpers.is_complex(a_dtype) and not helpers.is_complex(b_dtype):
            result = helpers.complex_ctr(out_dtype) + '(a.x * b, a.y * b)'
        elif not helpers.is_complex(a_dtype) and helpers.is_complex(b_dtype):
            result = helpers.complex_ctr(out_dtype) + '(b.x * a, b.y * a)'
        else:
            result = helpers.complex_ctr(out_dtype) + '(a.x * b.x - a.y * b.y', 'a.x * b.y + a.y * b.x)'
    %>

    return ${result};
}
</%def>


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
