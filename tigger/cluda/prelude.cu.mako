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
    WITHIN_KERNEL ${tp} operator+(${tp} a, ${tp} b)
    {
        return COMPLEX_CTR(${tp})(a.x + b.x, a.y + b.y);
    }
    WITHIN_KERNEL ${tp} operator-(${tp} a, ${tp} b)
    {
        return COMPLEX_CTR(${tp})(a.x - b.x, a.y - b.y);
    }
    WITHIN_KERNEL ${tp} operator+(${tp} a) { return a; }
    WITHIN_KERNEL ${tp} operator-(${tp} a) { return COMPLEX_CTR(${tp})(-a.x, -a.y); }
%endfor
%endif

<%def name="define_mul_func(name, a_dtype, b_dtype, out_dtype)">
WITHIN_KERNEL ${helpers.ctype(out_dtype)} ${name}(
    ${helpers.ctype(a_dtype)} a, ${helpers.ctype(b_dtype)} b)
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
