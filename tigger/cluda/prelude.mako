%if api == 'cuda':
    // taken from pycuda._cluda
    #define LOCAL_BARRIER __syncthreads()

    #define WITHIN_KERNEL __device__
    #define KERNEL extern "C" __global__
    #define GLOBAL_MEM /* empty */
    #define LOCAL_MEM __shared__
    #define LOCAL_MEM_DYNAMIC extern __shared__
    #define LOCAL_MEM_ARG /* empty */
    #define INLINE __forceinline__

    #define LID_FLAT threadIdx.x
    #define GID_FLAT (blockIdx.x + gridDim.x * blockIdx.y)
    #define ID_FLAT (LID_FLAT + blockDim.x * GID_FLAT)

    #define LID_0 threadIdx.x
    #define LID_1 threadIdx.y
    #define LID_2 threadIdx.z

    #define GID_0 blockIdx.x
    #define GID_1 blockIdx.y
    #define GID_2 blockIdx.z

    #define LSIZE_0 blockDim.x
    #define LSIZE_1 blockDim.y
    #define LSIZE_2 blockDim.z

%elif api == 'ocl':
    // taken from pyopencl._cluda
    #define LOCAL_BARRIER barrier(CLK_LOCAL_MEM_FENCE)

    #define WITHIN_KERNEL /* empty */
    #define KERNEL __kernel
    #define GLOBAL_MEM __global
    #define LOCAL_MEM __local
    #define LOCAL_MEM_DYNAMIC __local
    #define LOCAL_MEM_ARG __local
    #define INLINE inline

    #define LID_FLAT get_local_id(0)
    #define GID_FLAT get_group_id(0)
    #define ID_FLAT get_global_id(0)

    #define LID_0 get_local_id(0)
    #define LID_1 get_local_id(1)
    #define LID_2 get_local_id(2)

    #define GID_0 get_group_id(0)
    #define GID_1 get_group_id(1)
    #define GID_2 get_group_id(2)

    #define LSIZE_0 get_local_size(0)
    #define LSIZE_1 get_local_size(1)
    #define LSIZE_2 get_local_size(2)

    #if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64: enable
    #endif

%endif

%if api == 'cuda':
    #define COMPLEX_CTR(T) make_##T
%elif api == 'ocl':
    #define COMPLEX_CTR(T) (T)
%endif

## These operators are supported by OpenCL
%if api == 'cuda':
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
