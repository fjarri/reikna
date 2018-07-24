<%def name="prelude()">

%if compile_fast_math:
#define COMPILE_FAST_MATH
%endif

%if api == 'cuda':
    #define CUDA
    // taken from pycuda._cluda
    #define LOCAL_BARRIER __syncthreads()

    #define WITHIN_KERNEL __device__
    #define KERNEL extern "C" __global__
    #define GLOBAL_MEM /* empty */
    #define GLOBAL_MEM_ARG /* empty */
    #define LOCAL_MEM __shared__
    #define LOCAL_MEM_DYNAMIC extern __shared__
    #define LOCAL_MEM_ARG /* empty */
    #define CONSTANT_MEM __constant__
    #define CONSTANT_MEM_ARG /* empty */
    #define INLINE __forceinline__
    #define SIZE_T int
    #define VSIZE_T int

    // used to align fields in structures
    #define ALIGN(bytes) __align__(bytes)

    <%
        dimnames = ['x', 'y', 'z']
    %>

    WITHIN_KERNEL SIZE_T get_local_id(unsigned int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return threadIdx.${dimnames[n]};
    %endfor
        return 0;
    }

    WITHIN_KERNEL SIZE_T get_group_id(unsigned int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return blockIdx.${dimnames[n]};
    %endfor
        return 0;
    }

    WITHIN_KERNEL SIZE_T get_local_size(unsigned int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return blockDim.${dimnames[n]};
    %endfor
        return 1;
    }

    WITHIN_KERNEL SIZE_T get_num_groups(unsigned int dim)
    {
    %for n in range(3):
        if(dim == ${n}) return gridDim.${dimnames[n]};
    %endfor
        return 1;
    }

    WITHIN_KERNEL SIZE_T get_global_size(unsigned int dim)
    {
        return get_num_groups(dim) * get_local_size(dim);
    }

    WITHIN_KERNEL SIZE_T get_global_id(unsigned int dim)
    {
        return get_local_id(dim) + get_group_id(dim) * get_local_size(dim);
    }

%elif api == 'ocl':
    // taken from pyopencl._cluda
    #define LOCAL_BARRIER barrier(CLK_LOCAL_MEM_FENCE)

    // 'static' helps to avoid the "no previous prototype for function" warning
    #if __OPENCL_VERSION__ >= 120
    #define WITHIN_KERNEL static
    #else
    #define WITHIN_KERNEL
    #endif

    #define KERNEL __kernel
    #define GLOBAL_MEM __global
    #define GLOBAL_MEM_ARG __global
    #define LOCAL_MEM __local
    #define LOCAL_MEM_DYNAMIC __local
    #define LOCAL_MEM_ARG __local
    #define CONSTANT_MEM __constant
    #define CONSTANT_MEM_ARG __constant
    // INLINE is already defined in Beignet driver
    #ifndef INLINE
    #define INLINE inline
    #endif
    #define SIZE_T size_t
    #define VSIZE_T size_t

    // used to align fields in structures
    #define ALIGN(bytes) __attribute__ ((aligned(bytes)))

    #if defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64: enable
    #endif

%endif

%if constant_arrays is not None:
%for name in sorted(constant_arrays):
<%
    length, dtype = constant_arrays[name]
%>
CONSTANT_MEM ${dtypes.ctype(dtype)} ${name}[${length}];
%endfor
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

</%def>
